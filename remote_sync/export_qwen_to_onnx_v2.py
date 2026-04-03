#!/usr/bin/env python3
"""
Export Qwen model to ONNX format for ACL inference
"""
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

# Configuration
MODEL_PATH = "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen2.5-7B-Instruct"
ONNX_OUTPUT_DIR = "/home/HwHiAiUser/ICT/qwen_onnx_v2"
BATCH_SIZE = 1
SEQ_LENGTH = 512  # Reduced from 2048 for memory efficiency

def export_qwen_onnx():
    print(f"Loading model from {MODEL_PATH}...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True
    )
    model.eval()
    
    print("Model loaded. Preparing dummy inputs...")
    
    # Create dummy inputs
    dummy_input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)
    dummy_attention_mask = torch.ones((BATCH_SIZE, SEQ_LENGTH), dtype=torch.long)
    dummy_position_ids = torch.arange(SEQ_LENGTH, dtype=torch.long).unsqueeze(0).expand(BATCH_SIZE, -1)
    
    # Create output directory
    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(ONNX_OUTPUT_DIR, "qwen_model.onnx")
    
    print(f"Exporting to ONNX: {output_path}")
    print(f"Input shapes: input_ids={dummy_input_ids.shape}, attention_mask={dummy_attention_mask.shape}")
    
    try:
        # Export to ONNX
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
        
        print(f"✓ ONNX export successful: {output_path}")
        print(f"✓ File size: {os.path.getsize(output_path) / 1e9:.2f} GB")
        
        # Save tokenizer
        tokenizer.save_pretrained(ONNX_OUTPUT_DIR)
        print(f"✓ Tokenizer saved to {ONNX_OUTPUT_DIR}")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    export_qwen_onnx()
