import os
import sys
from pathlib import Path
from optimum.exporters.onnx import main_export

model_id = "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen2.5-1.5B-Instruct"
output_dir = Path("/home/HwHiAiUser/ICT/qwen_onnx")

print(f"Exporting model from {model_id} to {output_dir}...")

try:
    main_export(
        model_name_or_path=model_id,
        output=output_dir,
        task="text-generation",
        device="cpu",
        no_post_process=False,
    )
    print("Export complete.")
except Exception as e:
    print(f"Export failed: {e}")
