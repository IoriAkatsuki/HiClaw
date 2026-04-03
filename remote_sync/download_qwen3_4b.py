#!/usr/bin/env python3
"""
Download Qwen3-4B - The correct model for ACL deployment
"""
from huggingface_hub import snapshot_download
import os
import sys

# Qwen3-4B base model
model_id = "Qwen/Qwen3-4B"
output_dir = "/home/HwHiAiUser/ICT/models/qwen3-4b"

os.makedirs(output_dir, exist_ok=True)

print("="*60)
print("Downloading Qwen3-4B Model")
print("="*60)
print(f"Model ID: {model_id}")
print(f"Destination: {output_dir}")
print("="*60)
print("\nQwen3-4B is suitable for:")
print("  - 24GB memory board")
print("  - ACL inference with FP16")
print("  - Good balance of performance and size")
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
        f.write(f"Parameters: 4B\n")

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
