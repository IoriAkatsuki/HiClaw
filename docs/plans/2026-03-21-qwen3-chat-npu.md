# Qwen3-0.6B NPU Chat: No-Think + KV Cache + WebUI

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn the validated Qwen3-0.6B .om inference into a usable chat service with no-think mode, KV cache optimization, and a web chat UI integrated into the existing GUI.

**Architecture:** Two .om models (prefill + decode) with explicit KV cache tensor I/O eliminate O(n²) re-computation. A chat tab is added to the existing unified WebUI (port 8002). The backend uses SSE for streamed token delivery.

**Tech Stack:** ACL (Ascend), ONNX + ATC, Python HTTP server, vanilla JS, SSE

---

## KV Cache Specs (Qwen3-0.6B)

```
28 layers × 2 (key + value) = 56 KV cache tensors
Per tensor: [batch=1, n_kv_heads=8, seq_len, head_dim=128]
dtype: float16
Per-token-per-layer: 8 × 128 × 2 bytes = 2 KB
Per-token all layers: 2 KB × 28 × 2 = 112 KB
Max 512 tokens → 56 MB total KV cache
```

---

### Task 1: No-Think Mode

**Files:**
- Modify: `edge/qwen3_chat/chat_service.py`

**Step 1: Add no-think system prompt and `<think>` stripping**

In `ChatSession.__init__`, change `SYSTEM_PROMPT` and add post-processing:

```python
SYSTEM_PROMPT = "You are a helpful assistant. /no_think"
```

In `ChatSession.generate`, after decoding, strip `<think>...</think>` blocks:

```python
import re
reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
```

**Step 2: Test on board**

```bash
ssh HwHiAiUser@ict.local "source ~/miniconda3/Ascend/cann/set_env.sh && cd ~/ICT/edge/qwen3_chat && python3 -c \"
from chat_service import ChatSession
s = ChatSession(max_new_tokens=64, temperature=0.0)
reply, n, speed = s.generate('What is 2+3?')
print(f'Reply: {reply}')
print(f'Has think tag: {\"<think>\" in reply}')
s.close()
\""
```

Expected: Reply contains `5` or `2+3=5`, no `<think>` tags.

**Step 3: Commit**

```bash
git add edge/qwen3_chat/chat_service.py
git commit -m "feat(qwen3-chat): enable no-think mode, strip <think> blocks"
```

---

### Task 2: Export Prefill + Decode Models with KV Cache

**Files:**
- Create: `tools/export_qwen3_kvcache.py`

**Step 1: Write export script**

Two models:
- **Prefill**: `input_ids:[1,N]` → `logits:[1,N,V]` + 56 × `present.L.key/value:[1,8,N,128]`
- **Decode**: `input_ids:[1,1]` + 56 × `past.L.key/value:[1,8,P,128]` → `logits:[1,1,V]` + 56 × `present.L.key/value:[1,8,P+1,128]`

Key: wrap model to explicitly return KV cache as flat tensors.

```python
class PrefillWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=True)
        result = [out.logits]
        for kv in out.past_key_values:
            result.append(kv[0])  # key [1, 8, seq, 128]
            result.append(kv[1])  # value [1, 8, seq, 128]
        return tuple(result)


class DecodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, *past_kv_flat):
        # Reconstruct past_key_values from flat args
        past_kv = []
        for i in range(self.n_layers):
            k = past_kv_flat[2 * i]
            v = past_kv_flat[2 * i + 1]
            past_kv.append((k, v))
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         past_key_values=past_kv, use_cache=True)
        result = [out.logits]
        for kv in out.past_key_values:
            result.append(kv[0])
            result.append(kv[1])
        return tuple(result)
```

Export with `torch.onnx.export(dynamo=False, opset_version=17)`.

For ATC: prefill with `--input_shape="input_ids:1,512;attention_mask:1,512"`, decode with `--input_shape="input_ids:1,1;attention_mask:1,513;past.0.key:1,8,512,128;..."` using `--dynamic_dims` for the past seq_len dimension.

**Step 2: Run export locally (conda env)**

```bash
conda run -n cann850-qwen35-nightly python3 tools/export_qwen3_kvcache.py
```

**Step 3: ATC convert both models**

```bash
# Prefill
atc --model=qwen3_prefill.onnx --framework=5 --output=qwen3_prefill \
    --soc_version=Ascend310B1 --input_shape="input_ids:1,512;attention_mask:1,512" \
    --precision_mode=allow_fp32_to_fp16

# Decode (dynamic past_seq_len via --dynamic_dims)
atc --model=qwen3_decode.onnx --framework=5 --output=qwen3_decode \
    --soc_version=Ascend310B1 \
    --input_shape="input_ids:1,1;attention_mask:1,-1;past.0.key:1,8,-1,128;..." \
    --dynamic_dims="1~512;1~512;..." \
    --precision_mode=allow_fp32_to_fp16
```

NOTE: If ATC `--dynamic_dims` doesn't work for 310B1, fall back to fixed-length decode (pad KV cache to max_len=512).

**Step 4: Verify prefill .om produces identical logits to current seq512 model**

**Step 5: Commit**

```bash
git add tools/export_qwen3_kvcache.py
git commit -m "feat(qwen3): export prefill+decode models with KV cache"
```

---

### Task 3: KV Cache ACL Backend

**Files:**
- Modify: `edge/qwen3_chat/acl_backend.py`

**Step 1: Add `KVCacheModel` class**

```python
class KVCacheModel:
    """Dual-model backend: prefill + decode with explicit KV cache."""

    def __init__(self, prefill_om: str, decode_om: str, device_id=0):
        ...

    def prefill(self, token_ids: list[int]) -> tuple[np.ndarray, list[np.ndarray]]:
        """Returns (logits_last, kv_cache_list)."""
        ...

    def decode_step(self, token_id: int, kv_cache: list[np.ndarray],
                    seq_pos: int) -> tuple[np.ndarray, list[np.ndarray]]:
        """Returns (logits, updated_kv_cache)."""
        ...
```

KV cache tensors stay on device memory between steps (no host↔device copy per step).

**Step 2: Update `ChatSession` to use `KVCacheModel` when available**

Falls back to `OmModel` if prefill/decode .om not found.

**Step 3: Test decode speed**

Expected: ~3-5 tok/s (vs 0.3 tok/s without KV cache) — 10× improvement.

**Step 4: Commit**

```bash
git add edge/qwen3_chat/acl_backend.py edge/qwen3_chat/chat_service.py
git commit -m "feat(qwen3-chat): KV cache dual-model backend"
```

---

### Task 4: Chat WebUI Tab

**Files:**
- Modify: `webui_http_unified/index.html` (add Chat tab)
- Modify: `edge/unified_app/webui_server.py` (add chat API endpoints)

**Step 1: Add chat API endpoints to webui_server.py**

```python
# POST /api/chat/send   body: {"message": "..."}
#   → SSE stream: data: {"token": "word"}\n\n ... data: {"done": true, "stats": {...}}\n\n
# POST /api/chat/reset  → reset conversation history
# GET  /api/chat/history → current messages
```

The chat backend runs in a background thread. `ChatSession` is initialized once on first `/api/chat/send`.

**Step 2: Add Chat tab to index.html**

Add 6th tab "Chat" to the sidebar tab bar. Chat UI layout:

```
┌─────────────────────────┐
│  Message history        │
│  (scrollable)           │
│                         │
│  User: ...              │
│  Bot: ...               │
│  User: ...              │
│  Bot: ... (streaming)   │
├─────────────────────────┤
│ [input box]    [Send]   │
│ [Reset] [Greedy toggle] │
└─────────────────────────┘
```

Styling: match existing dark theme with cyan accent (#00f2ff).

Token streaming via SSE (`EventSource`). Show token count and speed after each reply.

**Step 3: Test in browser**

Open `http://ict.local:8002`, click Chat tab, send message, verify streamed response.

**Step 4: Commit**

```bash
git add webui_http_unified/index.html edge/unified_app/webui_server.py
git commit -m "feat(webui): add Chat tab with Qwen3-0.6B NPU inference"
```

---

### Task 5: Integration Test

**Files:**
- Create: `tests/test_qwen3_chat_service.py`

**Step 1: Write tests**

```python
class TestChatServiceNoThink(unittest.TestCase):
    """Test no-think mode strips <think> tags."""

    def test_strip_think_tags(self):
        from edge.qwen3_chat.chat_service import ChatSession
        text = "<think>\nLet me think...\n</think>\nThe answer is 5."
        import re
        result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        self.assertEqual(result, "The answer is 5.")

    def test_system_prompt_has_no_think(self):
        self.assertIn("/no_think", ChatSession.SYSTEM_PROMPT)
```

**Step 2: Run tests**

```bash
python3 -m unittest tests/test_qwen3_chat_service.py -v
```

**Step 3: Commit**

```bash
git add tests/test_qwen3_chat_service.py
git commit -m "test: add Qwen3 chat service unit tests"
```

---

## Execution Order

| Task | Depends On | Risk |
|------|------------|------|
| 1. No-Think Mode | None | Low — prompt change only |
| 2. Export KV Cache Models | None | **High** — torch export + ATC with 56 KV tensors may fail |
| 3. KV Cache Backend | Task 2 | Medium — ACL buffer management |
| 4. Chat WebUI Tab | Task 1 | Medium — frontend + SSE |
| 5. Integration Test | Task 1 | Low |

**Critical path**: Task 1 → Task 4 gives a usable (slow) chat UI immediately. Task 2 → Task 3 is the performance optimization that can be done in parallel.

---

## Fallback

If KV cache export/ATC fails (Task 2), the system still works with O(n²) prefill-only mode at ~0.3 tok/s. The WebUI (Task 4) works regardless — it just streams slower.
