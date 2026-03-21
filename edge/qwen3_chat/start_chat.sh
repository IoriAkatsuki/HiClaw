#!/bin/bash
# Start Qwen3-0.6B CLI chat on Ascend 310B1
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source ~/miniconda3/Ascend/cann/set_env.sh
cd "$SCRIPT_DIR"
exec python3 chat_service.py "$@"
