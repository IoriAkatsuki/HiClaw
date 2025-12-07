#!/usr/bin/env bash
# 使用 ATC 将 MindIR 转 OM
# 示例：bash convert_to_ascend_om.sh route_a_yolov5n.mindir route_a_yolov5n.om

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "用法: $0 <input.mindir> <output.om> [img_size]" >&2
  exit 1
fi

MINDIR=$1
OUTPUT=$2
IMG_SIZE=${3:-640}

atc \
  --model="${MINDIR}" \
  --framework=1 \
  --output="${OUTPUT%.*}" \
  --input_shape="images:1,3,${IMG_SIZE},${IMG_SIZE}" \
  --soc_version=Ascend310B1 \
  --log=info

echo "[DONE] OM 已输出到 ${OUTPUT}"
