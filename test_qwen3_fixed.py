#!/usr/bin/env python3
"""
Test script for fixed Qwen3 ACL Inference
"""
import sys
sys.path.insert(0, '/home/HwHiAiUser/ICT')

from qwen3_acl_inference import Qwen3ACLInference

MODEL_PATH = "/home/HwHiAiUser/ICT/qwen3_fp16.om"
TOKENIZER_PATH = "/home/HwHiAiUser/ICT/models/qwen3-8b"

print("="*60)
print("Testing Fixed Qwen3 Inference Engine")
print("="*60)

try:
    print("\n[1/3] Initializing engine...")
    engine = Qwen3ACLInference(MODEL_PATH, TOKENIZER_PATH, max_seq_len=256)

    print("\n[2/3] Testing text generation...")
    test_prompt = "你好，请介绍一下你自己"
    print(f"Prompt: {test_prompt}")

    response, time_taken = engine.generate_text(
        test_prompt,
        max_new_tokens=50
    )

    print("\n[3/3] Results:")
    print("="*60)
    print(f"Response: {response}")
    print(f"Time: {time_taken:.2f}s")
    print(f"Response length: {len(response)} chars")
    print("="*60)

    # Check if response is valid
    if len(response) > 10 and not any(ord(c) > 0xFFFF for c in response):
        print("\n✓ SUCCESS: Generated complete text without garbled characters!")
    else:
        print("\n✗ FAILED: Response is too short or contains garbled text")

    engine.cleanup()

except Exception as e:
    print(f"\n✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
