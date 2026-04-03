#!/usr/bin/env python3
"""
Export Qwen3-8B to ONNX format for ACL inference on Ascend 310B
"""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Configuration
MODEL_PATH = "/home/HwHiAiUser/ICT/models/qwen3-8b"
ONNX_OUTPUT_DIR = "/home/HwHiAiUser/ICT/qwen3_onnx"
BATCH_SIZE = 1
SEQ_LENGTH = 512  # Reduced for memory efficiency on 24GB board

def export_qwen3_onnx():
    print(f"Loading Qwen3-8B model from {MODEL_PATH}...")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,  # Use FP16 to fit in 24GB memory
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()

    print("✓ Model loaded successfully")
    print(f"  - Model dtype: {model.dtype}")
    print(f"  - Device: {model.device}")

    # Create dummy inputs for ONNX export
    print(f"\nPreparing dummy inputs (batch={BATCH_SIZE}, seq_len={SEQ_LENGTH})...")
    dummy_input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)
    dummy_attention_mask = torch.ones((BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)
    dummy_position_ids = torch.arange(SEQ_LENGTH, dtype=torch.long).unsqueeze(0).expand(BATCH_SIZE, -1)

    # Create output directory
    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(ONNX_OUTPUT_DIR, "qwen3_model.onnx")

    print(f"\nExporting to ONNX: {output_path}")
    print(f"  Input shapes:")
    print(f"    - input_ids: {dummy_input_ids.shape}")
    print(f"    - attention_mask: {dummy_attention_mask.shape}")
    print(f"    - position_ids: {dummy_position_ids.shape}")

    try:
        # Export to ONNX
        print("\nStarting ONNX export (this may take several minutes)...")
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask, dummy_position_ids),
            output_path,
            input_names=["input_ids", "attention_mask", "position_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "position_ids": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=13,
            do_constant_folding=True,
            verbose=False
        )

        file_size = os.path.getsize(output_path) / (1024**3)
        print(f"\n{'='*60}")
        print(f"✓ ONNX export successful!")
        print(f"  - Output: {output_path}")
        print(f"  - Size: {file_size:.2f} GB")
        print(f"{'='*60}")

        # Save tokenizer
        tokenizer.save_pretrained(ONNX_OUTPUT_DIR)
        print(f"\n✓ Tokenizer saved to {ONNX_OUTPUT_DIR}")

        # Save export info
        with open(os.path.join(ONNX_OUTPUT_DIR, "EXPORT_INFO.txt"), "w") as f:
            f.write(f"Model: Qwen3-8B\n")
            f.write(f"Source: {MODEL_PATH}\n")
            f.write(f"ONNX: {output_path}\n")
            f.write(f"Batch size: {BATCH_SIZE}\n")
            f.write(f"Sequence length: {SEQ_LENGTH}\n")
            f.write(f"Precision: FP16\n")

        return output_path

    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ONNX export failed!")
        print(f"Error: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = export_qwen3_onnx()
    if result:
        print("\n✓ Export process completed successfully!")
    else:
        print("\n✗ Export process failed!")
        exit(1)
