"""Qwen3-0.6B chat service on Ascend 310B1 via .om + ACL.

Uses prefill-only model (no KV cache). Each generation step re-runs
the full sequence through the model (O(n^2) but simple).
"""
import re, time, sys
import numpy as np
from pathlib import Path

DEFAULT_OM       = str(Path.home() / "ICT/models/qwen3_0.6b_seq512.om")
DEFAULT_PREFILL  = str(Path.home() / "ICT/models/qwen3_prefill.om")
DEFAULT_DECODE   = str(Path.home() / "ICT/models/qwen3_decode.om")
DEFAULT_TOK      = str(Path.home() / "ICT/models/qwen3-0.6b")


def _sample_top_p(logits: np.ndarray, temperature: float = 0.7,
                   top_p: float = 0.9) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    logits -= logits.max()
    probs = np.exp(logits) / np.exp(logits).sum()
    sorted_idx = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cum, top_p) + 1
    candidates = sorted_idx[:cutoff]
    p = probs[candidates]
    p /= p.sum()
    return int(np.random.choice(candidates, p=p))


class ChatSession:
    """Manages a multi-turn chat with Qwen3-0.6B .om backend."""

    SYSTEM_PROMPT = "You are a helpful assistant. /no_think"
    ROLE_MAP = {"system": "system", "user": "user", "assistant": "assistant"}

    def __init__(self, om_path: str = DEFAULT_OM, tok_dir: str = DEFAULT_TOK,
                 max_new_tokens: int = 128, temperature: float = 0.7,
                 top_p: float = 0.9,
                 prefill_om: str = DEFAULT_PREFILL,
                 decode_om: str = DEFAULT_DECODE):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

        self._use_kv = Path(prefill_om).exists() and Path(decode_om).exists()
        if self._use_kv:
            from acl_backend import KVCacheModel
            self.model = KVCacheModel(prefill_om, decode_om)
            self.max_seq = self.model.max_seq
            print(f"[ChatSession] KV cache mode (prefill+decode), max_seq={self.max_seq}")
        else:
            from acl_backend import OmModel
            self.model = OmModel(om_path)
            self.max_seq = self.model.seq_len
            print(f"[ChatSession] Prefill-only mode, max_seq={self.max_seq}")

        self.max_new = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.messages: list[dict] = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        self.eos_ids = set()
        for name in ("eos_token_id",):
            v = getattr(self.tok, name, None)
            if isinstance(v, int):
                self.eos_ids.add(v)
            elif isinstance(v, (list, tuple)):
                self.eos_ids.update(v)
        if not self.eos_ids:
            self.eos_ids = {151643, 151645}  # Qwen3 default

    def _build_prompt_ids(self, messages: list[dict]) -> list[int]:
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        return self.tok.encode(text)

    def generate(self, user_msg: str) -> str:
        self.messages.append({"role": "user", "content": user_msg})
        prompt_ids = self._build_prompt_ids(self.messages)

        if len(prompt_ids) > self.max_seq - 4:
            prompt_ids = prompt_ids[-(self.max_seq - 4):]

        generated: list[int] = []
        t0 = time.time()

        if self._use_kv:
            generated = self._generate_kv(prompt_ids)
        else:
            generated = self._generate_prefill_only(prompt_ids)

        elapsed = time.time() - t0
        n = len(generated)
        speed = n / elapsed if elapsed > 0 else 0

        reply = self.tok.decode(generated, skip_special_tokens=True)
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply, n, speed

    def _generate_prefill_only(self, prompt_ids: list[int]) -> list[int]:
        generated = []
        for _ in range(self.max_new):
            all_ids = prompt_ids + generated
            if len(all_ids) >= self.max_seq:
                break
            logits = self.model.forward(all_ids)
            tok_id = _sample_top_p(logits[-1], self.temperature, self.top_p)
            if tok_id in self.eos_ids:
                break
            generated.append(tok_id)
        return generated

    def _generate_kv(self, prompt_ids: list[int]) -> list[int]:
        self.model.reset_kv()
        logits = self.model.prefill(prompt_ids)
        generated = []
        for _ in range(self.max_new):
            if self.model._kv_pos + len(generated) >= self.max_seq:
                break
            tok_id = _sample_top_p(logits, self.temperature, self.top_p)
            if tok_id in self.eos_ids:
                break
            generated.append(tok_id)
            logits = self.model.decode_step(tok_id)
        return generated

    def close(self):
        self.model.close()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Qwen3-0.6B CLI chat on Ascend 310B1")
    p.add_argument("--om", default=DEFAULT_OM)
    p.add_argument("--tok", default=DEFAULT_TOK)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()

    temp = 0.0 if args.greedy else args.temperature
    session = ChatSession(args.om, args.tok, args.max_tokens, temp)

    print(f"Qwen3-0.6B on NPU (seq_len={session.max_seq}). Type 'quit' to exit.\n")
    try:
        while True:
            user = input("You: ").strip()
            if not user or user.lower() in ("quit", "exit", "q"):
                break
            reply, n_tok, speed = session.generate(user)
            print(f"Bot: {reply}")
            print(f"  [{n_tok} tokens, {speed:.1f} tok/s]\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        session.close()
        print("\nBye.")


if __name__ == "__main__":
    main()
