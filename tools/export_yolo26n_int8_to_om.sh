#!/usr/bin/env bash
# 将 INT8 ONNX 转换为 Ascend OM（适用于 Ascend 310B1）。
# 用法：
#   ./export_yolo26n_int8_to_om.sh [INT8_ONNX] [OUTPUT_OM]
# 或通过环境变量覆盖：
#   INT8_ONNX=/path/to/int8.onnx OM_PATH=/path/to/out ./export_yolo26n_int8_to_om.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/models/route_a_yolo26}"
INT8_ONNX="${1:-${INT8_ONNX:-$OUTPUT_DIR/yolo26n_aug_full_8419_gpu_int8.onnx}}"
OM_PATH="${2:-${OM_PATH:-$OUTPUT_DIR/yolo26n_aug_full_8419_gpu_int8}}"
IMG_SIZE="${IMG_SIZE:-640}"
ATC_INPUT_NAME="${ATC_INPUT_NAME:-images}"
SOC_VERSION="${SOC_VERSION:-Ascend310B1}"

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

if ! command -v atc >/dev/null 2>&1; then
    echo "[ERROR] 未找到 atc，请先 source set_env.sh 或在装有 CANN 的环境中执行"
    exit 1
fi

if [ ! -f "$INT8_ONNX" ]; then
    echo "[ERROR] INT8 ONNX 不存在: $INT8_ONNX"
    echo "        请先运行 quantize_yolo26n_amct.py"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[ATC] INT8 ONNX → OM"
echo "      input  : $INT8_ONNX"
echo "      output : ${OM_PATH}.om"
echo "      shape  : $ATC_INPUT_NAME:1,3,$IMG_SIZE,$IMG_SIZE"
echo "      soc    : $SOC_VERSION"
echo ""

atc \
    --model="$INT8_ONNX" \
    --framework=5 \
    --output="$OM_PATH" \
    --input_shape="$ATC_INPUT_NAME:1,3,$IMG_SIZE,$IMG_SIZE" \
    --soc_version="$SOC_VERSION" \
    --log=info

echo ""
echo "[DONE] OM 已生成: ${OM_PATH}.om"
