#!/bin/bash

# Ascend ATC Conversion Script for Qwen ONNX to OM

set -e

# Environment setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Paths
ONNX_MODEL="/home/HwHiAiUser/ICT/qwen_onnx_v2/qwen_model.onnx"
OM_OUTPUT="/home/HwHiAiUser/ICT/qwen2.5-7b_fp16.om"
SOC_VERSION="Ascend310B1"  # OrangePi AIpro uses 310B

echo "========================================"
echo "Converting ONNX to OM for Ascend NPU"
echo "========================================"
echo "Input ONNX: $ONNX_MODEL"
echo "Output OM: $OM_OUTPUT"
echo "SOC Version: $SOC_VERSION"
echo "========================================"

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
echo "Conversion complete!"
echo "Output: $OM_OUTPUT"
ls -lh "$OM_OUTPUT"
