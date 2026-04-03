#!/usr/bin/env python3
"""
Export Qwen3-4B using Optimum (More reliable than torch.onnx.export)
"""
import os
import sys
from pathlib import Path

MODEL_PATH = "/home/HwHiAiUser/ICT/models/qwen3-4b"
ONNX_OUTPUT_DIR = "/home/HwHiAiUser/ICT/qwen3_4b_onnx"

print("="*60)
print("Qwen3-4B ONNX Export via Optimum")
print("="*60)
print(f"Model: {MODEL_PATH}")
print(f"Output: {ONNX_OUTPUT_DIR}")
print("="*60)

# Check model exists
if not os.path.exists(MODEL_PATH):
    print(f"✗ Model not found: {MODEL_PATH}")
    sys.exit(1)

os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)

print("\n[1/2] Loading Optimum exporter...")
try:
    from optimum.exporters.onnx import main_export
    print("✓ Optimum loaded")
except ImportError as e:
    print(f"✗ Failed to import optimum: {e}")
    sys.exit(1)

print("\n[2/2] Exporting to ONNX...")
print("This will take 10-20 minutes for a 4B model...")
print("")

try:
    # Export using Optimum - handles complex models better
    main_export(
        model_name_or_path=MODEL_PATH,
        output=Path(ONNX_OUTPUT_DIR),
        task="text-generation",  # Don't use -with-past to avoid KV cache complexity
        device="cpu",
        opset=14,
        trust_remote_code=True,
        no_post_process=True,  # Skip validation to save time
    )

    print("\n" + "="*60)
    print("✓ Export successful!")
    print("="*60)

    # List files
    print("\nGenerated files:")
    for f in os.listdir(ONNX_OUTPUT_DIR):
        fpath = os.path.join(ONNX_OUTPUT_DIR, f)
        if os.path.isfile(fpath):
            size_gb = os.path.getsize(fpath) / (1024**3)
            print(f"  - {f}: {size_gb:.2f} GB")

except Exception as e:
    print("\n" + "="*60)
    print("✗ Export failed!")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
