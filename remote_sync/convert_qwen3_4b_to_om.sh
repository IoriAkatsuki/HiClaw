#!/bin/bash
#
# Convert Qwen3-4B ONNX to OM format with input shapes
#

set -e

# Setup Ascend environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Paths
ONNX_MODEL="/home/HwHiAiUser/ICT/qwen3_4b_onnx/model.onnx"
OM_OUTPUT="/home/HwHiAiUser/ICT/qwen3_4b_fp16.om"
SOC_VERSION="Ascend310B1"

# Input shapes (batch=1, seq_len=512)
INPUT_SHAPE="input_ids:1,512;attention_mask:1,512"

echo "========================================"
echo "Converting Qwen3-4B ONNX to OM"
echo "========================================"
echo "Input ONNX: $ONNX_MODEL"
echo "Output OM: $OM_OUTPUT"
echo "SOC: $SOC_VERSION"
echo "Input Shape: $INPUT_SHAPE"
echo "========================================"

# Check ONNX exists
if [ ! -f "$ONNX_MODEL" ]; then
    echo "ERROR: ONNX file not found!"
    exit 1
fi

echo ""
echo "Starting ATC conversion..."
echo "This may take 10-30 minutes..."
echo ""

# ATC conversion with input shapes
atc \
  --model="$ONNX_MODEL" \
  --framework=5 \
  --output="${OM_OUTPUT%.om}" \
  --input_shape="$INPUT_SHAPE" \
  --soc_version="$SOC_VERSION" \
  --input_format=ND \
  --log=error \
  --precision_mode=allow_fp32_to_fp16 \
  --op_select_implmode=high_performance \
  --enable_small_channel=1 \
  2>&1 | tee atc_conversion.log

if [ -f "$OM_OUTPUT" ]; then
    echo ""
    echo "========================================"
    echo "✓ Conversion successful!"
    echo "========================================"
    ls -lh "$OM_OUTPUT"
    echo "========================================"
else
    echo ""
    echo "========================================"
    echo "✗ Conversion failed!"
    echo "========================================"
    echo "Check atc_conversion.log for details"
    tail -50 atc_conversion.log
    exit 1
fi
