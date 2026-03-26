"""Qwen3 chat service on Ascend 310B1 — multi-backend (NPU .om / CPU llama.cpp)."""
import json, re, time, subprocess, sys
import numpy as np
from pathlib import Path

MODELS_DIR  = Path.home() / "ICT/models"
LLAMA_CLI   = Path.home() / "llama.cpp/build/bin/llama-cli"

BACKENDS = {
    "npu_qwen3": {
        "label": "Qwen3-0.6B NPU (prefill-only, 0.3 tok/s)",
        "type": "npu",
        "om": str(MODELS_DIR / "qwen3_0.6b_seq512.om"),
        "tok": str(MODELS_DIR / "qwen3-0.6b"),
    },
    "cpu_qwen3": {
        "label": "Qwen3-0.6B CPU Q4 (7.4 tok/s)",
        "type": "llama_cpp",
        "gguf": str(MODELS_DIR / "gguf/Qwen3-0.6B-Q4_K_M.gguf"),
        "threads": 3,
    },
    "cpu_qwen35": {
        "label": "Qwen3.5-0.8B CPU Q4 (~1 tok/s)",
        "type": "llama_cpp",
        "gguf": str(MODELS_DIR / "gguf/Qwen3.5-0.8B-Q4_K_M.gguf"),
        "threads": 3,
    },
}
DEFAULT_BACKEND = "cpu_qwen3"


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


class NpuBackend:
    """ACL .om prefill-only backend."""

    def __init__(self, om_path: str, tok_dir: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from acl_backend import OmModel
        self.model = OmModel(om_path)
        self.max_seq = self.model.seq_len

    def generate(self, messages: list[dict], max_new: int, temperature: float,
                 top_p: float) -> tuple[str, int, float]:
        text = self.tok.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        prompt_ids = self.tok.encode(text)
        if len(prompt_ids) > self.max_seq - 4:
            prompt_ids = prompt_ids[-(self.max_seq - 4):]

        eos_ids = {151643, 151645}
        generated = []
        t0 = time.time()
        for _ in range(max_new):
            all_ids = prompt_ids + generated
            if len(all_ids) >= self.max_seq:
                break
            logits = self.model.forward(all_ids)
            tok_id = _sample_top_p(logits[-1], temperature, top_p)
            if tok_id in eos_ids:
                break
            generated.append(tok_id)

        elapsed = time.time() - t0
        n = len(generated)
        speed = n / elapsed if elapsed > 0 else 0
        reply = self.tok.decode(generated, skip_special_tokens=True)
        return reply, n, speed

    def close(self):
        self.model.close()


class LlamaCppBackend:
    """llama.cpp server-mode backend with KV cache."""

    def __init__(self, gguf_path: str, threads: int = 3, port: int = 8090):
        self.gguf = gguf_path
        self.threads = threads
        self.port = port
        self._proc = None
        self._start_server()

    def _start_server(self):
        cmd = [
            str(LLAMA_CLI).replace("llama-cli", "llama-server"),
            "-m", self.gguf,
            "-t", str(self.threads),
            "-ngl", "0",
            # 在 Ascend 环境里显式禁用设备 offload，避免 llama.cpp 误走 ggml-cann。
            "--device", "none",
            "--port", str(self.port),
            "--host", "127.0.0.1",
            "--reasoning-budget", "0",
            "-c", "2048",
        ]
        server_bin = Path(cmd[0])
        if not server_bin.exists():
            raise FileNotFoundError(f"llama-server not found: {server_bin}")
        self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
        # Wait for server startup
        import urllib.request
        for _ in range(30):
            time.sleep(1)
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health",
                                       timeout=2)
                print(f"[LlamaCpp] Server ready on port {self.port}", flush=True)
                return
            except Exception:
                if self._proc.poll() is not None:
                    raise RuntimeError(f"llama-server exited: {self._proc.returncode}")
        raise RuntimeError("llama-server startup timeout")

    def generate(self, messages: list[dict], max_new: int, temperature: float,
                 top_p: float) -> tuple[str, int, float]:
        import urllib.request
        payload = json.dumps({
            "messages": messages,
            "max_tokens": max_new,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
        }).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
        elapsed = time.time() - t0

        reply = body["choices"][0]["message"]["content"]
        n = body.get("usage", {}).get("completion_tokens", len(reply.split()))
        speed = n / elapsed if elapsed > 0 else 0
        return reply, n, speed

    def close(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=5)


class ChatSession:
    """Multi-turn chat with switchable backend."""

    SYSTEM_PROMPT = "You are a helpful assistant. /no_think"

    def __init__(self, backend_name: str = DEFAULT_BACKEND,
                 max_new_tokens: int = 128, temperature: float = 0.7,
                 top_p: float = 0.9):
        self.backend_name = backend_name
        self.max_new = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.messages: list[dict] = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]
        self._backend = None
        self._init_backend(backend_name)

    def _init_backend(self, name: str):
        if self._backend:
            self._backend.close()
            self._backend = None
        cfg = BACKENDS.get(name)
        if not cfg:
            raise ValueError(f"Unknown backend: {name}. Available: {list(BACKENDS)}")
        if cfg["type"] == "npu":
            self._backend = NpuBackend(cfg["om"], cfg["tok"])
        elif cfg["type"] == "llama_cpp":
            self._backend = LlamaCppBackend(cfg["gguf"], cfg.get("threads", 3))
        self.backend_name = name
        print(f"[ChatSession] Backend: {cfg['label']}", flush=True)

    def switch_backend(self, name: str):
        self._init_backend(name)

    def generate(self, user_msg: str) -> tuple[str, int, float]:
        self.messages.append({"role": "user", "content": user_msg})
        reply, n, speed = self._backend.generate(
            self.messages, self.max_new, self.temperature, self.top_p
        )
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
        self.messages.append({"role": "assistant", "content": reply})
        return reply, n, speed

    def reset(self):
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

    def close(self):
        if self._backend:
            self._backend.close()

    @staticmethod
    def list_backends() -> dict:
        return {k: v["label"] for k, v in BACKENDS.items()}


def main():
    import argparse
    p = argparse.ArgumentParser(description="Qwen3 CLI chat on Ascend 310B1")
    p.add_argument("--backend", default=DEFAULT_BACKEND, choices=BACKENDS.keys())
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--greedy", action="store_true")
    args = p.parse_args()

    temp = 0.0 if args.greedy else args.temperature
    session = ChatSession(args.backend, args.max_tokens, temp)

    print(f"\nBackends: {json.dumps(ChatSession.list_backends(), indent=2)}")
    print(f"Active: {session.backend_name}")
    print("Commands: /switch <name>, /reset, /quit\n")

    try:
        while True:
            user = input("You: ").strip()
            if not user:
                continue
            if user.lower() in ("/quit", "quit", "exit", "q"):
                break
            if user.startswith("/switch "):
                name = user.split(None, 1)[1].strip()
                try:
                    session.switch_backend(name)
                    print(f"Switched to: {name}\n")
                except Exception as e:
                    print(f"Switch failed: {e}\n")
                continue
            if user == "/reset":
                session.reset()
                print("Conversation reset.\n")
                continue
            reply, n_tok, speed = session.generate(user)
            print(f"Bot: {reply}")
            print(f"  [{n_tok} tokens, {speed:.1f} tok/s, backend={session.backend_name}]\n")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        session.close()
        print("\nBye.")


if __name__ == "__main__":
    main()
