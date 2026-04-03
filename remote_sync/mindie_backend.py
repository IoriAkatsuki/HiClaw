#!/usr/bin/env python3
"""
MindIE HTTP 后端适配层（OpenAI-like API）。

说明：
1. 本模块用于 WebUI 灰度切换，不依赖 ACL。
2. 默认按 OpenAI Chat Completions 风格请求后端服务。
3. 对 llama.cpp / MindIE 风格 SSE 做真流式解析，减少首屏等待。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Generator, Optional, Sequence
from urllib import error, request


@dataclass
class _GenCfg:
    max_new_tokens: int = 128
    temperature: float = 0.9
    top_k: int = 40
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[Sequence[int]] = None


class MindIEBackend:
    def __init__(self, endpoint: str, model: str, timeout_s: int = 120):
        if not endpoint:
            raise ValueError("MindIE endpoint 不能为空")
        self.endpoint = endpoint
        self.model = model
        self.timeout_s = int(timeout_s)
        self.last_generation_stats: Dict[str, float] = {}

    def _normalize_cfg(self, config=None, **legacy_kwargs) -> _GenCfg:
        cfg = _GenCfg()
        if config is not None:
            for key in cfg.__dict__.keys():
                if hasattr(config, key):
                    setattr(cfg, key, getattr(config, key))
        for key, value in legacy_kwargs.items():
            if value is not None and hasattr(cfg, key):
                setattr(cfg, key, value)
        cfg.max_new_tokens = max(1, int(cfg.max_new_tokens))
        cfg.temperature = float(cfg.temperature)
        cfg.top_k = max(0, int(cfg.top_k))
        cfg.top_p = min(1.0, max(0.0, float(cfg.top_p)))
        cfg.repetition_penalty = max(1.0, float(cfg.repetition_penalty))
        return cfg

    def _build_payload(self, prompt: str, cfg: _GenCfg, stream: bool) -> Dict[str, object]:
        return {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "stream": stream,
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "none",
            "top_k": cfg.top_k,
            "do_sample": cfg.do_sample,
            "repetition_penalty": cfg.repetition_penalty,
        }

    def _open(self, payload: Dict[str, object]):
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return request.urlopen(req, timeout=self.timeout_s)
        except error.HTTPError as exc:
            msg = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
            raise RuntimeError(f"MindIE HTTP 错误: {exc.code}, {msg}") from exc
        except Exception as exc:
            raise RuntimeError(f"MindIE 请求失败: {exc}") from exc

    @staticmethod
    def _extract_text(body: Dict[str, object]) -> str:
        choices = body.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, dict):
            return ""
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if content not in (None, ""):
                return str(content)
            reasoning = message.get("reasoning_content")
            if reasoning not in (None, ""):
                return str(reasoning)
        text = choice.get("text")
        if text not in (None, ""):
            return str(text)
        return ""

    @staticmethod
    def _extract_delta(body: Dict[str, object]) -> str:
        choices = body.get("choices", [])
        if not choices:
            return ""
        choice = choices[0]
        if not isinstance(choice, dict):
            return ""
        delta = choice.get("delta")
        if isinstance(delta, dict):
            content = delta.get("content")
            if content:
                return str(content)
            reasoning = delta.get("reasoning_content")
            if reasoning:
                return str(reasoning)
        text = choice.get("text")
        if text:
            return str(text)
        return ""

    @staticmethod
    def _iter_sse_payloads(resp) -> Generator[str, None, None]:
        event_lines = []

        def flush_event():
            data_parts = []
            for item in event_lines:
                if not item or item.startswith(":") or not item.startswith("data:"):
                    continue
                data_parts.append(item[5:].lstrip())
            if not data_parts:
                return None
            payload = "\n".join(data_parts).strip()
            return payload or None

        def should_flush_before_append(next_line: str) -> bool:
            if not event_lines or not next_line.startswith("data:"):
                return False
            payload = flush_event()
            if payload in (None, "", "[DONE]"):
                return payload == "[DONE]"
            try:
                json.loads(payload)
                return True
            except json.JSONDecodeError:
                return False

        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
            if not line:
                payload = flush_event()
                if payload is not None:
                    yield payload
                event_lines = []
                continue
            if should_flush_before_append(line):
                payload = flush_event()
                if payload is not None:
                    yield payload
                event_lines = []
            event_lines.append(line)

        payload = flush_event()
        if payload is not None:
            yield payload

    def _record_stats(
        self,
        *,
        elapsed: float,
        completion_tokens: float,
        first_token_ms: float = 0.0,
        timings: Optional[Dict[str, float]] = None,
    ) -> None:
        timings = timings or {}
        predicted_per_second = float(timings.get("predicted_per_second", 0.0) or 0.0)
        predicted_per_token_ms = float(timings.get("predicted_per_token_ms", 0.0) or 0.0)
        prompt_ms = float(timings.get("prompt_ms", 0.0) or 0.0)
        self.last_generation_stats = {
            "total_time_s": elapsed,
            "generated_tokens": completion_tokens,
            "tokens_per_sec": predicted_per_second if predicted_per_second > 0 else (completion_tokens / elapsed if elapsed > 0 else 0.0),
            "first_token_ms": first_token_ms,
            "avg_token_ms": predicted_per_token_ms if predicted_per_token_ms > 0 else ((elapsed * 1000.0 / completion_tokens) if completion_tokens > 0 else 0.0),
            "prompt_ms": prompt_ms,
            "avg_h2d_ms": 0.0,
            "avg_execute_ms": 0.0,
            "avg_d2h_ms": 0.0,
        }

    def generate_text(self, prompt: str, config=None, **legacy_kwargs):
        cfg = self._normalize_cfg(config=config, **legacy_kwargs)
        payload = self._build_payload(prompt, cfg, stream=False)

        start = time.perf_counter()
        with self._open(payload) as resp:
            raw = resp.read().decode("utf-8")

        elapsed = time.perf_counter() - start
        body = json.loads(raw)
        text = self._extract_text(body)
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        completion_tokens = float(usage.get("completion_tokens", 0)) if isinstance(usage, dict) else 0.0
        if completion_tokens <= 0:
            completion_tokens = float(max(1, len(text)))
        self._record_stats(
            elapsed=elapsed,
            completion_tokens=completion_tokens,
            timings=body.get("timings") if isinstance(body.get("timings"), dict) else None,
        )
        return text, elapsed

    def stream_generate(self, prompt: str, config=None, **legacy_kwargs) -> Generator[Dict[str, object], None, None]:
        cfg = self._normalize_cfg(config=config, **legacy_kwargs)
        payload = self._build_payload(prompt, cfg, stream=True)

        start = time.perf_counter()
        first_token_at = None
        final_timings: Dict[str, float] = {}
        text = ""
        step = 0

        with self._open(payload) as resp:
            for data in self._iter_sse_payloads(resp):
                if data == "[DONE]":
                    break

                try:
                    body = json.loads(data)
                except json.JSONDecodeError:
                    continue
                timings = body.get("timings")
                if isinstance(timings, dict):
                    final_timings = timings

                delta = self._extract_delta(body)
                if not delta:
                    continue

                if first_token_at is None:
                    first_token_at = time.perf_counter()
                step += 1
                text += delta
                yield {
                    "delta": delta,
                    "text": text,
                    "step": step,
                    "token_id": -1,
                    "step_ms": 0.0,
                }

        elapsed = time.perf_counter() - start
        completion_tokens = float(final_timings.get("predicted_n", 0) or 0)
        if completion_tokens <= 0:
            completion_tokens = float(max(1, step))
        first_token_ms = 0.0 if first_token_at is None else (first_token_at - start) * 1000.0
        self._record_stats(
            elapsed=elapsed,
            completion_tokens=completion_tokens,
            first_token_ms=first_token_ms,
            timings=final_timings,
        )
