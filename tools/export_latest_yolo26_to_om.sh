#!/usr/bin/env bash
# 导出 2026_3_12 最新 yolo26 权重到 ONNX，再转换为 Ascend 310B OM。

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_WEIGHTS="$ROOT_DIR/2026_3_12/runs/train/yolo26n_aug_full_8419_gpu/weights/best.pt"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/models/route_a_yolo26}"
MODEL_BASENAME="${MODEL_BASENAME:-yolo26n_aug_full_8419_gpu}"
WEIGHTS_PATH="${WEIGHTS_PATH:-$DEFAULT_WEIGHTS}"
ONNX_PATH="${ONNX_PATH:-$OUTPUT_DIR/$MODEL_BASENAME.onnx}"
OM_PATH="${OM_PATH:-$OUTPUT_DIR/$MODEL_BASENAME.om}"
IMG_SIZE="${IMG_SIZE:-640}"
ATC_INPUT_NAME="${ATC_INPUT_NAME:-images}"
SOC_VERSION="${SOC_VERSION:-Ascend310B1}"
EXPORT_DEVICE="${EXPORT_DEVICE:-cpu}"

mkdir -p "$OUTPUT_DIR"

echo "[1/2] 导出 ONNX"
python3 "$ROOT_DIR/tools/export_latest_yolo26_onnx.py" \
  --weights "$WEIGHTS_PATH" \
  --output "$ONNX_PATH" \
  --imgsz "$IMG_SIZE" \
  --device "$EXPORT_DEVICE" \
  --simplify

if ! command -v atc >/dev/null 2>&1; then
  echo ""
  echo "[WARN] 未找到 atc，已完成 ONNX 导出。"
  echo "       请在已安装 CANN/ATC 的环境中继续执行："
  echo "       atc --model=\"$ONNX_PATH\" --framework=5 --output=\"${OM_PATH%.om}\" --input_shape=\"$ATC_INPUT_NAME:1,3,$IMG_SIZE,$IMG_SIZE\" --soc_version=$SOC_VERSION"
  exit 0
fi

echo ""
echo "[2/2] 转换 OM"
atc \
  --model="$ONNX_PATH" \
  --framework=5 \
  --output="${OM_PATH%.om}" \
  --input_shape="$ATC_INPUT_NAME:1,3,$IMG_SIZE,$IMG_SIZE" \
  --soc_version="$SOC_VERSION" \
  --log=info

echo ""
echo "[DONE] OM 已生成: $OM_PATH"
