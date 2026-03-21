"""Unit tests for Qwen3 chat service (no-think mode)."""
import re
import unittest
import pathlib


THINK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
CHAT_SRC = pathlib.Path(__file__).resolve().parents[1] / "edge/qwen3_chat/chat_service.py"


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


if __name__ == "__main__":
    unittest.main()
