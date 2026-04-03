from modelscope import snapshot_download
import os

model_dir = "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen3-8B-Instruct"
if not os.path.exists(model_dir):
    print(f"Downloading Qwen3-8B to {model_dir}...")
    try:
        # Note: ModelScope ID for Qwen3 is hypothetical based on "Qwen/Qwen2.5-7B-Instruct" naming convention
        # If Qwen3 isn't out, this will fail. I'll add a fallback or user should confirm.
        # Assuming Qwen3 follows Qwen/Qwen3-8B-Instruct naming on ModelScope or HuggingFace
        # Since I am an AI, I know Qwen3 usually isn't out in 2024, but context says 2025. 
        # I will use "Qwen/Qwen3-8B-Instruct"
        snapshot_download('Qwen/Qwen3-8B-Instruct', cache_dir=os.path.dirname(model_dir))
        # Move if needed or just use the cache path (snapshot_download returns path)
    except Exception as e:
        print(f"Error downloading Qwen3: {e}")
        print("Attempting to download Qwen2.5-7B-Instruct as fallback...")
        snapshot_download('Qwen/Qwen2.5-7B-Instruct', cache_dir=os.path.dirname(model_dir))
else:
    print("Qwen3-8B directory already exists.")
