#!/usr/bin/env python3
"""Debug Qwen3 token generation"""
import sys
sys.path.insert(0, '/home/HwHiAiUser/ICT')

from qwen3_acl_inference import Qwen3ACLInference
import numpy as np

MODEL_PATH = "/home/HwHiAiUser/ICT/qwen3_1_7b_fp16.om"
TOKENIZER_PATH = "/home/HwHiAiUser/ICT/models/qwen3-1.7b"

print("="*60)
print("Token Generation Debug")
print("="*60)

engine = Qwen3ACLInference(MODEL_PATH, TOKENIZER_PATH, max_seq_len=256)

# Test prompt
prompt = "你好"
print(f"\nPrompt: '{prompt}'")

# Encode
prompt_ids = engine.tokenizer.encode(prompt, add_special_tokens=True)
print(f"Prompt token IDs: {prompt_ids}")
print(f"Prompt tokens: {[engine.tokenizer.decode([tid]) for tid in prompt_ids]}")

# Prepare input
import numpy as np
input_ids = np.zeros((1, 256), dtype=np.int64)
attention_mask = np.zeros((1, 256), dtype=np.int64)
position_ids = np.arange(256, dtype=np.int64).reshape(1, -1)

seq_len = len(prompt_ids)
input_ids[0, :seq_len] = prompt_ids
attention_mask[0, :seq_len] = 1

print(f"\nRunning single inference step...")
next_token_id, logits = engine._inference_single_step(input_ids, attention_mask, position_ids)

print(f"\nNext token ID: {next_token_id}")
print(f"Next token text: '{engine.tokenizer.decode([next_token_id])}'")
print(f"Logits shape: {logits.shape}")
print(f"Top 10 token IDs: {np.argsort(logits)[-10:][::-1]}")
print(f"Top 10 tokens: {[engine.tokenizer.decode([tid]) for tid in np.argsort(logits)[-10:][::-1]]}")
print(f"Top 10 logits: {np.sort(logits)[-10:][::-1]}")

# Check special tokens
print(f"\nSpecial tokens:")
print(f"  EOS: {engine.tokenizer.eos_token_id} = '{engine.tokenizer.eos_token}'")
print(f"  PAD: {engine.tokenizer.pad_token_id if hasattr(engine.tokenizer, 'pad_token_id') else 'N/A'}")
print(f"  BOS: {engine.tokenizer.bos_token_id if hasattr(engine.tokenizer, 'bos_token_id') else 'N/A'}")

engine.cleanup()
