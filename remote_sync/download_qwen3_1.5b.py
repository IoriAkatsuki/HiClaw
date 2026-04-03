#!/usr/bin/env python3
"""
Download Qwen3-1.5B - A smaller model suitable for 24GB memory boards
"""
from huggingface_hub import snapshot_download
import os
import sys

# Qwen3-1.5B is much more suitable for the 24GB board
model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # Using 2.5 as Qwen3-1.5B might not exist yet
output_dir = "/home/HwHiAiUser/ICT/models/qwen-1.5b"

os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("Downloading Qwen 1.5B Model")
print("="*60)
print(f"Model ID: {model_id}")
print(f"Destination: {output_dir}")
print("="*60)
print("\nThis model is ~3GB, much more suitable for:")
print("  - ONNX export on 24GB board")
print("  - ACL inference with FP16")
print("  - Real-time inference")
print("")

try:
    downloaded_path = snapshot_download(
        repo_id=model_id,
        cache_dir="/home/HwHiAiUser/.cache/huggingface",
        local_dir=output_dir,
        resume_download=True
    )

    print("\n" + "="*60)
    print(f"✓ SUCCESS! Downloaded {model_id}")
    print(f"✓ Location: {downloaded_path}")
    print("="*60)

    # Save model info
    with open(os.path.join(output_dir, "MODEL_INFO.txt"), "w") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Path: {downloaded_path}\n")
        f.write(f"Size: ~3GB\n")
        f.write(f"Parameters: 1.5B\n")

    # Check file size
    import subprocess
    result = subprocess.run(['du', '-sh', output_dir], capture_output=True, text=True)
    print(f"\nTotal size: {result.stdout.split()[0]}")

    print("\n✓ Download complete!")

except Exception as e:
    print(f"\n✗ ERROR: Failed to download {model_id}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
