"""Qwen3 chat service — thin wrapper around llama.cpp server (CANN backend)."""
import re
import json
import logging
import urllib.request
import urllib.error

log = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant. /no_think"

DEFAULT_BASE_URL = "http://127.0.0.1:8080"


class ChatSession:
    """Stateful multi-turn chat session backed by a llama.cpp /v1/chat/completions endpoint."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 max_tokens: int = 256,
                 temperature: float = 0.6,
                 top_p: float = 0.95):
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.history: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

    def generate(self, user_msg: str) -> str:
        """Send *user_msg*, return assistant reply (think-tags stripped)."""
        self.history.append({"role": "user", "content": user_msg})

        payload = json.dumps({
            "messages": self.history,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            log.error("llama.cpp server unreachable: %s", exc)
            raise

        reply = body["choices"][0]["message"]["content"]
        reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()

        self.history.append({"role": "assistant", "content": reply})
        return reply

    def reset(self) -> None:
        """Clear conversation history, keep system prompt."""
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
