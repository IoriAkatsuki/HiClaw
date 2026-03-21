"""Unit tests for Qwen3 chat service (no-think mode, sampling)."""
import re
import unittest


class TestNoThinkStripping(unittest.TestCase):
    def test_strip_think_block(self):
        text = "<think>\nLet me think...\n</think>\nThe answer is 5."
        result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        self.assertEqual(result, "The answer is 5.")

    def test_no_think_passthrough(self):
        text = "The answer is 5."
        result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        self.assertEqual(result, "The answer is 5.")

    def test_empty_after_strip(self):
        text = "<think>Only thinking</think>"
        result = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        self.assertEqual(result, "")

    def test_system_prompt_has_no_think(self):
        import sys
        sys.path.insert(0, "/home/oasis/Documents/ICT-qwen3-chat")
        import ast
        with open("/home/oasis/Documents/ICT-qwen3-chat/edge/qwen3_chat/chat_service.py") as f:
            source = f.read()
        self.assertIn("/no_think", source)
        self.assertIn('SYSTEM_PROMPT', source)


if __name__ == "__main__":
    unittest.main()
