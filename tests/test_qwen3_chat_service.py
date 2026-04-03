"""Unit tests for Qwen3 chat service (no-think mode)."""
import importlib.util
import pathlib
import re
import subprocess
import unittest
from unittest import mock


THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
CHAT_SRC = pathlib.Path(__file__).resolve().parents[1] / "edge/qwen3_chat/chat_service.py"


def load_chat_module():
    spec = importlib.util.spec_from_file_location("qwen3_chat_service_test", CHAT_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNoThinkStripping(unittest.TestCase):
    def test_strip_think_block(self):
        text = "<think>\nLet me think...\n</think>\nThe answer is 5."
        self.assertEqual(THINK_PATTERN.sub("", text).strip(), "The answer is 5.")

    def test_no_think_passthrough(self):
        text = "The answer is 5."
        self.assertEqual(THINK_PATTERN.sub("", text).strip(), "The answer is 5.")

    def test_empty_after_strip(self):
        text = "<think>Only thinking</think>"
        self.assertEqual(THINK_PATTERN.sub("", text).strip(), "")

    def test_multiple_think_blocks(self):
        text = "<think>a</think>Hello <think>b</think>world"
        self.assertEqual(THINK_PATTERN.sub("", text).strip(), "Hello world")

    def test_system_prompt_source(self):
        code = CHAT_SRC.read_text()
        self.assertIn("/no_think", code)
        self.assertIn("SYSTEM_PROMPT", code)
        self.assertIn('re.sub(r"<think>.*?</think>"', code)


class TestLlamaCppBackend(unittest.TestCase):
    def test_start_server_command_disables_device_offload(self):
        module = load_chat_module()
        module.LLAMA_CLI = pathlib.Path("/tmp/llama-cli")

        proc = mock.Mock()
        proc.poll.return_value = None

        response = mock.Mock()
        response.__enter__ = mock.Mock(return_value=response)
        response.__exit__ = mock.Mock(return_value=False)

        with mock.patch.object(module.Path, "exists", return_value=True), \
             mock.patch.object(module.subprocess, "check_output", side_effect=module.subprocess.CalledProcessError(1, "ss")), \
             mock.patch.object(module.subprocess, "Popen", return_value=proc) as popen_mock, \
             mock.patch.object(module.time, "sleep", return_value=None), \
             mock.patch("urllib.request.urlopen", return_value=response):
            backend = module.LlamaCppBackend("model.gguf", threads=2, port=9001)

        cmd = popen_mock.call_args.args[0]
        self.assertIn("--device", cmd)
        self.assertIn("none", cmd)
        self.assertEqual(cmd[0], "/tmp/llama-server")
        self.assertEqual(cmd[cmd.index("--port") + 1], "9001")
        backend.close()

    def test_close_kills_process_after_wait_timeout(self):
        module = load_chat_module()
        backend = module.LlamaCppBackend.__new__(module.LlamaCppBackend)
        backend._proc = mock.Mock()
        backend._proc.poll.return_value = None
        backend._proc.wait.side_effect = [
            subprocess.TimeoutExpired(cmd="llama-server", timeout=5),
            None,
        ]

        backend.close()

        backend._proc.terminate.assert_called_once()
        backend._proc.kill.assert_called_once()
        self.assertEqual(backend._proc.wait.call_count, 2)


if __name__ == "__main__":
    unittest.main()
