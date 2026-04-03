#!/usr/bin/env python3
"""
Test ACL inference with existing qwen_fp16.om file
"""
import sys
import os

# Add Ascend Python path
sys.path.insert(0, '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages')

import acl
import numpy as np
from transformers import AutoTokenizer
import time

class QwenACLTest:
    def __init__(self, model_path, tokenizer_path, device_id=0):
        self.device_id = device_id
        self.model_path = model_path

        print(f"[Init] Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None

        self.init_acl()

    def check_ret(self, ret, message):
        if ret != 0:
            raise Exception(f"{message} failed with ret={ret}")

    def init_acl(self):
        print("[ACL] Initializing...")
        ret = acl.init()
        self.check_ret(ret, "acl.init")

        ret = acl.rt.set_device(self.device_id)
        self.check_ret(ret, "acl.rt.set_device")

        self.context, ret = acl.rt.create_context(self.device_id)
        self.check_ret(ret, "acl.rt.create_context")

        self.stream, ret = acl.rt.create_stream()
        self.check_ret(ret, "acl.rt.create_stream")

        print(f"[ACL] Loading model: {self.model_path}")
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        self.check_ret(ret, "acl.mdl.load_from_file")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        self.check_ret(ret, "acl.mdl.get_desc")

        input_num = acl.mdl.get_num_inputs(self.model_desc)
        output_num = acl.mdl.get_num_outputs(self.model_desc)

        print(f"[ACL] ✓ Model loaded successfully")
        print(f"      - Inputs: {input_num}")
        print(f"      - Outputs: {output_num}")

        # Print input/output details
        for i in range(input_num):
            size = acl.mdl.get_input_size_by_index(self.model_desc, i)
            print(f"      - Input[{i}] size: {size} bytes")

        for i in range(output_num):
            size = acl.mdl.get_output_size_by_index(self.model_desc, i)
            print(f"      - Output[{i}] size: {size} bytes")

    def cleanup(self):
        print("\n[ACL] Cleaning up...")
        if self.model_id:
            acl.mdl.unload(self.model_id)
        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("[ACL] ✓ Cleanup complete")

def main():
    MODEL_PATH = "/home/HwHiAiUser/ICT/qwen_fp16.om"

    # Try multiple possible tokenizer paths
    tokenizer_paths = [
        "/home/HwHiAiUser/ICT/models/qwen3-4b",
        "/home/HwHiAiUser/ICT/models/qwen3-8b",
        "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen2.5-1.5B-Instruct",
    ]

    TOKENIZER_PATH = None
    for path in tokenizer_paths:
        if os.path.exists(path):
            TOKENIZER_PATH = path
            break

    if not TOKENIZER_PATH:
        print("✗ No tokenizer found!")
        return

    print("="*60)
    print("ACL Inference Test with Existing OM File")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    print("="*60)

    try:
        engine = QwenACLTest(MODEL_PATH, TOKENIZER_PATH)
        print("\n✓ ACL initialization successful!")
        print("\nThis confirms:")
        print("  1. ACL library is working")
        print("  2. OM file can be loaded")
        print("  3. NPU device is accessible")

        engine.cleanup()

        print("\n" + "="*60)
        print("✓ Test completed successfully!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
