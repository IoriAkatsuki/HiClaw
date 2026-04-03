import json
import unittest
from unittest import mock

from remote_sync.mindie_backend import MindIEBackend


class _FakeResponse:
    def __init__(self, body=None, lines=None):
        self._body = body.encode("utf-8") if isinstance(body, str) else (body or b"")
        self._lines = [line.encode("utf-8") if isinstance(line, str) else line for line in (lines or [])]

    def read(self):
        return self._body

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class MindIEBackendTest(unittest.TestCase):
    def test_build_payload_disables_thinking(self):
        backend = MindIEBackend("http://127.0.0.1:1025/v1/chat/completions", "qwen")
        payload = backend._build_payload("你好", backend._normalize_cfg(), stream=True)
        self.assertTrue(payload["stream"])
        self.assertFalse(payload["enable_thinking"])
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(payload["reasoning_format"], "none")

    @mock.patch("remote_sync.mindie_backend.request.urlopen")
    def test_generate_text_prefers_message_content(self, mock_urlopen):
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "测试回复", "reasoning_content": "忽略"}}],
            "usage": {"completion_tokens": 4},
            "timings": {"predicted_per_second": 5.0, "predicted_per_token_ms": 200.0},
        })
        mock_urlopen.return_value = _FakeResponse(body=body)
        backend = MindIEBackend("http://127.0.0.1:1025/v1/chat/completions", "qwen")

        text, _ = backend.generate_text("你好")

        self.assertEqual(text, "测试回复")
        self.assertEqual(backend.last_generation_stats["generated_tokens"], 4.0)
        self.assertEqual(backend.last_generation_stats["tokens_per_sec"], 5.0)

    @mock.patch("remote_sync.mindie_backend.request.urlopen")
    def test_generate_text_skips_none_content_and_falls_back(self, mock_urlopen):
        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": None, "reasoning_content": "回退内容"}}],
            "usage": {"completion_tokens": 4},
        })
        mock_urlopen.return_value = _FakeResponse(body=body)
        backend = MindIEBackend("http://127.0.0.1:1025/v1/chat/completions", "qwen")

        text, _ = backend.generate_text("你好")

        self.assertEqual(text, "回退内容")

    @mock.patch("remote_sync.mindie_backend.request.urlopen")
    @mock.patch("remote_sync.mindie_backend.time.perf_counter")
    def test_stream_generate_parses_sse_and_updates_metrics(self, mock_perf_counter, mock_urlopen):
        mock_perf_counter.side_effect = [10.0, 11.2, 13.0]
        lines = [
            'data: {"choices":[{"delta":{"role":"assistant","content":null},"finish_reason":null}]}\n',
            'data: {"choices":[{"delta":{"content":"你好"},"finish_reason":null}]}\n',
            'data: {"choices":[{"delta":{"content":"，世界"},"finish_reason":null}]}\n',
            'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"predicted_n":2,"predicted_per_second":4.0,"predicted_per_token_ms":250.0,"prompt_ms":123.0}}\n',
            "data: [DONE]\n",
        ]
        mock_urlopen.return_value = _FakeResponse(lines=lines)
        backend = MindIEBackend("http://127.0.0.1:1025/v1/chat/completions", "qwen")

        chunks = list(backend.stream_generate("你好"))

        self.assertEqual([chunk["delta"] for chunk in chunks], ["你好", "，世界"])
        self.assertEqual(chunks[-1]["text"], "你好，世界")
        self.assertAlmostEqual(backend.last_generation_stats["first_token_ms"], 1200.0)
        self.assertEqual(backend.last_generation_stats["generated_tokens"], 2.0)
        self.assertEqual(backend.last_generation_stats["tokens_per_sec"], 4.0)
        self.assertEqual(backend.last_generation_stats["avg_token_ms"], 250.0)
        self.assertEqual(backend.last_generation_stats["prompt_ms"], 123.0)

    @mock.patch("remote_sync.mindie_backend.request.urlopen")
    @mock.patch("remote_sync.mindie_backend.time.perf_counter")
    def test_stream_generate_handles_heartbeat_multiline_and_text_fallback(self, mock_perf_counter, mock_urlopen):
        mock_perf_counter.side_effect = [1.0, 1.4, 2.0]
        lines = [
            ": keep-alive\n",
            "data:\n",
            "\n",
            'data: {"choices":\n',
            'data: [{"text":"你"}], "timings": {"predicted_n": 2}}\n',
            "\n",
            "data: {not-json}\n",
            "\n",
            'data: {"choices":[{"delta":{"content":"好"}}]}\n',
            "\n",
            "data: [DONE]\n",
            "\n",
        ]
        mock_urlopen.return_value = _FakeResponse(lines=lines)
        backend = MindIEBackend("http://127.0.0.1:1025/v1/chat/completions", "qwen")

        chunks = list(backend.stream_generate("你好"))

        self.assertEqual([chunk["delta"] for chunk in chunks], ["你", "好"])
        self.assertEqual(chunks[-1]["text"], "你好")
        self.assertAlmostEqual(backend.last_generation_stats["first_token_ms"], 400.0)
        self.assertEqual(backend.last_generation_stats["generated_tokens"], 2.0)
