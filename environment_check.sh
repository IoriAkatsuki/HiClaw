#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${1:-${MODEL_PATH:-}}"
IMAGE_PATH="${2:-${IMAGE_PATH:-$ROOT_DIR/current_frame.jpg}}"

ENV_CANDIDATES=(
  "/usr/local/Ascend/ascend-toolkit/set_env.sh"
  "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
)

SET_ENV=""
for candidate in "${ENV_CANDIDATES[@]}"; do
  if [[ -f "$candidate" ]]; then
    SET_ENV="$candidate"
    break
  fi
done

if [[ -n "$SET_ENV" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$SET_ENV"
  set -u
fi

echo "== 基础环境 =="
echo "HOST=$(hostname)"
echo "PWD=$(pwd)"
echo "DATE=$(date '+%F %T %Z')"
uname -a

echo
echo "== 命令检查 =="
for cmd in python3 pip3 npu-smi; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "$cmd: $(command -v "$cmd")"
  else
    echo "$cmd: MISSING"
  fi
done

echo
echo "== Ascend 环境 =="
if [[ -n "$SET_ENV" ]]; then
  echo "set_env.sh: $SET_ENV"
else
  echo "set_env.sh: MISSING"
fi

echo
echo "== 系统资源 =="
free -h || true

CANDIDATES=()
if [[ -n "$MODEL_PATH" ]]; then
  CANDIDATES+=("$MODEL_PATH")
fi
CANDIDATES+=(
  "$ROOT_DIR/models/qwen3.5-0.8b"
  "$ROOT_DIR/models/Qwen3.5-0.8B"
  "/home/HwHiAiUser/ICT/models/qwen3.5-0.8b"
  "/home/HwHiAiUser/ICT/models/Qwen3.5-0.8B"
  "/home/HwHiAiUser/ICT/models/Qwen/Qwen3.5-0.8B"
)

echo
echo "== 候选模型目录 =="
FOUND_MODEL=""
for path in "${CANDIDATES[@]}"; do
  [[ -z "$path" ]] && continue
  if [[ -d "$path" ]]; then
    echo "FOUND: $path"
    [[ -z "$FOUND_MODEL" ]] && FOUND_MODEL="$path"
  else
    echo "MISS:  $path"
  fi
done

if [[ -z "$FOUND_MODEL" ]]; then
  echo "WARN: 未找到默认候选模型目录，请显式传入模型路径。"
fi

echo
echo "== Python 检查 =="
python3 - "$FOUND_MODEL" "$IMAGE_PATH" <<'PY'
import json
import sys
from pathlib import Path

model_path = Path(sys.argv[1]) if sys.argv[1] else None
image_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
status = 0

mods = ["torch", "torch_npu", "transformers", "PIL", "huggingface_hub", "numpy", "decorator", "scipy", "attr", "psutil", "cloudpickle", "ml_dtypes", "tornado", "absl"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}\tOK\t{getattr(mod, '__version__', 'n/a')}")
    except Exception as exc:
        status = 1
        print(f"{name}\tERR\t{type(exc).__name__}: {exc}")

try:
    import torch
    print(f"torch.npu.exists\t{hasattr(torch, 'npu')}")
    if hasattr(torch, 'npu'):
        try:
            print(f"torch.npu.is_available\t{torch.npu.is_available()}")
        except Exception as exc:
            status = 1
            print(f"torch.npu.is_available\tERR\t{type(exc).__name__}: {exc}")
except Exception as exc:
    status = 1
    print(f"torch_probe\tERR\t{type(exc).__name__}: {exc}")

if image_path:
    if image_path.exists():
        print(f"image_path\tOK\t{image_path}")
    else:
        status = 1
        print(f"image_path\tERR\tmissing: {image_path}")

try:
    import numpy as np
    print(f"numpy_version\t{np.__version__}")
    major = int(str(np.__version__).split('.', 1)[0])
    if major >= 2:
        status = 1
        print(f"numpy_compat\tERR\tCANN/torch_npu 目前需要 numpy<2，当前为 {np.__version__}")
except Exception as exc:
    status = 1
    print(f"numpy_probe\tERR\t{type(exc).__name__}: {exc}")

if model_path and model_path.exists():
    try:
        config = json.loads((model_path / 'config.json').read_text())
        preproc = json.loads((model_path / 'preprocessor_config.json').read_text())
        print(f"model_type\t{config.get('model_type')}")
        print(f"architectures\t{config.get('architectures')}")
        print(f"processor_class\t{preproc.get('processor_class')}")
    except Exception as exc:
        status = 1
        print(f"model_probe\tERR\t{type(exc).__name__}: {exc}")
else:
    status = 1
    print(f"model_path\tERR\tmissing or unspecified: {model_path}")

sys.exit(status)
PY

PY_STATUS=$?

echo
echo "== npu-smi =="
if command -v npu-smi >/dev/null 2>&1; then
  npu-smi info || true
fi

exit $PY_STATUS
