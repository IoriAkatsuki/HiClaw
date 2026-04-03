#!/bin/bash
# ATC Conversion Script for Qwen3-8B ONNX to OM

set -e

# Environment setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Paths
ONNX_MODEL="/home/HwHiAiUser/ICT/qwen3_onnx/qwen3_model.onnx"
OM_OUTPUT="/home/HwHiAiUser/ICT/qwen3_fp16.om"
SOC_VERSION="Ascend310B1"  # OrangePi AIpro uses Ascend 310B

echo "========================================"
echo "Converting Qwen3-8B ONNX to OM"
echo "========================================"
echo "Input ONNX: $ONNX_MODEL"
echo "Output OM: $OM_OUTPUT"
echo "SOC Version: $SOC_VERSION"
echo "========================================"

# Check if ONNX file exists
if [ ! -f "$ONNX_MODEL" ]; then
    echo "ERROR: ONNX model not found at $ONNX_MODEL"
    exit 1
fi

# ATC conversion command
atc \
  --model="$ONNX_MODEL" \
  --framework=5 \
  --output="${OM_OUTPUT%.om}" \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --input_shape="input_ids:1,512;attention_mask:1,512;position_ids:1,512" \
  --precision_mode=allow_fp32_to_fp16 \
  --op_select_implmode=high_performance \
  --optypelist_for_implmode="Gelu" \
  --log=error \
  --enable_small_channel=1

echo ""
echo "========================================"
echo "✓ Conversion complete!"
echo "========================================"
echo "Output: $OM_OUTPUT"
ls -lh "$OM_OUTPUT"
echo "========================================"
