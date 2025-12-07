#!/usr/bin/env python3
"""Quick test for Qwen3 inference"""
import sys
sys.path.insert(0, '/home/HwHiAiUser/ICT')

print("Testing Qwen3 Inference...")

try:
    from qwen3_acl_inference import Qwen3ACLInference

    MODEL_PATH = "/home/HwHiAiUser/ICT/qwen3_1_7b_fp16.om"
    TOKENIZER_PATH = "/home/HwHiAiUser/ICT/models/qwen3-1.7b"

    print(f"Loading model from: {MODEL_PATH}")
    print(f"Loading tokenizer from: {TOKENIZER_PATH}")

    engine = Qwen3ACLInference(MODEL_PATH, TOKENIZER_PATH, max_seq_len=512)

    prompt = "你好"
    print(f"\nPrompt: '{prompt}'")

    response, time_taken = engine.generate_text(prompt, max_new_tokens=20)

    print(f"\nResponse: '{response}'")
    print(f"Response bytes: {response.encode('utf-8')}")
    print(f"Response repr: {repr(response)}")
    print(f"Time: {time_taken:.2f}s")

    engine.cleanup()

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
