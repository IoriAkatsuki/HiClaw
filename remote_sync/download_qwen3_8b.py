#!/usr/bin/env python3
from huggingface_hub import snapshot_download
import os
import sys

# Qwen3-8B base model (not instruct)
model_id = "Qwen/Qwen3-8B"
output_dir = "/home/HwHiAiUser/ICT/models/qwen3-8b"

os.makedirs(output_dir, exist_ok=True)

print(f"Downloading {model_id}...")
print(f"Destination: {output_dir}")
print("="*60)

try:
    downloaded_path = snapshot_download(
        repo_id=model_id,
        cache_dir="/home/HwHiAiUser/.cache/huggingface",
        local_dir=output_dir,
        resume_download=True
    )

    print("\n" + "="*60)
    print(f"SUCCESS! Downloaded {model_id}")
    print(f"Location: {downloaded_path}")
    print("="*60 + "\n")

    # Save model info
    with open(os.path.join(output_dir, "MODEL_INFO.txt"), "w") as f:
        f.write(f"Model: {model_id}\n")
        f.write(f"Path: {downloaded_path}\n")

    print("Download complete!")

except Exception as e:
    print(f"\nERROR: Failed to download {model_id}")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
