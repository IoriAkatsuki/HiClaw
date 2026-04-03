#!/usr/bin/env python3
"""
Export Qwen3-4B to ONNX using legacy exporter
"""
import torch
import os
import sys

# Force use of legacy ONNX exporter
os.environ['TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK'] = '0'

from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/home/HwHiAiUser/ICT/models/qwen3-4b"
ONNX_OUTPUT_DIR = "/home/HwHiAiUser/ICT/qwen3_4b_onnx"
BATCH_SIZE = 1
SEQ_LENGTH = 256  # Reduced to 256 for stability

def export_with_legacy():
    print("="*60)
    print("Qwen3-4B ONNX Export (Legacy Mode)")
    print("="*60)

    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,  # Use FP32 for ONNX export
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    print("✓ Model loaded")

    print("\n[2/4] Preparing inputs...")
    dummy_input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)
    dummy_attention_mask = torch.ones((BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)
    dummy_position_ids = torch.arange(SEQ_LENGTH, dtype=torch.long).unsqueeze(0)

    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(ONNX_OUTPUT_DIR, "qwen3_4b_model.onnx")

    print(f"\n[3/4] Exporting to ONNX...")
    print(f"Output: {output_path}")
    print("Using legacy exporter with reduced constraints...")

    try:
        # Disable dynamo to use legacy exporter
        with torch.no_grad():
            torch.onnx.export(
                model,
                (dummy_input_ids, dummy_attention_mask, dummy_position_ids),
                output_path,
                input_names=["input_ids", "attention_mask", "position_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {1: "sequence_length"},
                    "attention_mask": {1: "sequence_length"},
                    "position_ids": {1: "sequence_length"},
                    "logits": {1: "sequence_length"}
                },
                opset_version=14,
                do_constant_folding=False,  # Disable for compatibility
                verbose=False,
                export_params=True,
                # Use legacy exporter
                dynamo=False
            )

        size_gb = os.path.getsize(output_path) / (1024**3)
        print(f"\n✓ Export successful!")
        print(f"  Size: {size_gb:.2f} GB")

        print("\n[4/4] Saving tokenizer...")
        tokenizer.save_pretrained(ONNX_OUTPUT_DIR)
        print("✓ Complete!")

        return output_path

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = export_with_legacy()
    sys.exit(0 if result else 1)
