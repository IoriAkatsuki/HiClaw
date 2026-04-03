import os
from huggingface_hub import snapshot_download
import sys

model_id = "Qwen/Qwen2.5-7B-Instruct"  # Qwen3-8B may not be released yet, using 2.5-7B
output_dir = "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen2.5-7B-Instruct"

print(f"Downloading {model_id} to {output_dir}...")

try:
    # Create parent directory if needed
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    
    # Download from HuggingFace
    downloaded_path = snapshot_download(
        repo_id=model_id,
        cache_dir="/home/HwHiAiUser/.cache/huggingface",
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    
    print(f"Download complete! Model saved to: {downloaded_path}")
    
except Exception as e:
    print(f"Error downloading model: {e}")
    print("\nTrying to check if Qwen3-8B-Instruct exists...")
    
    # Try Qwen3 if available
    try:
        qwen3_id = "Qwen/Qwen3-8B-Instruct"
        qwen3_dir = "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen3-8B-Instruct"
        print(f"Attempting to download {qwen3_id}...")
        
        downloaded_path = snapshot_download(
            repo_id=qwen3_id,
            cache_dir="/home/HwHiAiUser/.cache/huggingface",
            local_dir=qwen3_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"Qwen3-8B downloaded to: {downloaded_path}")
    except Exception as e2:
        print(f"Qwen3 also failed: {e2}")
        sys.exit(1)
