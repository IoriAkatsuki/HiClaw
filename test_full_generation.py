#!/usr/bin/env python3
"""Test full generation with more tokens"""
import sys
sys.path.insert(0, '/home/HwHiAiUser/ICT')

from qwen3_acl_inference import Qwen3ACLInference

MODEL_PATH = "/home/HwHiAiUser/ICT/qwen3_1_7b_fp16.om"
TOKENIZER_PATH = "/home/HwHiAiUser/ICT/models/qwen3-1.7b"

print("="*60)
print("Full Generation Test (128 tokens)")
print("="*60)

engine = Qwen3ACLInference(MODEL_PATH, TOKENIZER_PATH, max_seq_len=512)

prompt = "你好，请介绍一下你自己"
print(f"\nPrompt: {prompt}")

response, time_taken = engine.generate_text(prompt, max_new_tokens=128)

print(f"\n{'='*60}")
print("RESPONSE:")
print("="*60)
print(response)
print("="*60)
print(f"\nGenerated {len(response)} characters in {time_taken:.1f}s")
print(f"Speed: {len(response)/time_taken:.1f} chars/sec")

engine.cleanup()
