#!/usr/bin/env python3
"""
Export Qwen3-8B to ONNX using Optimum
More reliable than torch.onnx.export for large transformer models
"""
import os
import sys

MODEL_PATH = "/home/HwHiAiUser/ICT/models/qwen3-8b"
ONNX_OUTPUT_DIR = "/home/HwHiAiUser/ICT/qwen3_onnx"

print("="*60)
print("Qwen3-8B ONNX Export (Using Optimum)")
print("="*60)
print(f"Model: {MODEL_PATH}")
print(f"Output: {ONNX_OUTPUT_DIR}")
print("="*60)

# Check if optimum is installed
try:
    from optimum.exporters.onnx import main_export
    print("✓ Optimum is installed")
except ImportError:
    print("✗ Optimum not installed. Installing...")
    os.system("pip3 install --user optimum[exporters]")
    from optimum.exporters.onnx import main_export
    print("✓ Optimum installed successfully")

os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)

print("\nStarting ONNX export...")
print("Note: This may take 10-20 minutes for an 8B model")
print("")

try:
    # Export using Optimum
    # This handles complex transformer architectures better than torch.onnx.export
    main_export(
        model_name_or_path=MODEL_PATH,
        output=ONNX_OUTPUT_DIR,
        task="text-generation-with-past",  # Use cached attention for better performance
        device="cpu",
        fp16=True,  # Export in FP16 for smaller size and faster inference
        opset=13,
        trust_remote_code=True
    )

    print("\n" + "="*60)
    print("✓ ONNX export successful!")
    print("="*60)

    # List generated files
    print("\nGenerated files:")
    for root, dirs, files in os.walk(ONNX_OUTPUT_DIR):
        for file in files:
            fpath = os.path.join(root, file)
            size = os.path.getsize(fpath) / (1024**3)
            print(f"  - {file}: {size:.2f} GB")

    print(f"\nOutput directory: {ONNX_OUTPUT_DIR}")

except Exception as e:
    print("\n" + "="*60)
    print("✗ Export failed!")
    print("="*60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
