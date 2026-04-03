#!/bin/bash
# 推送本地改动到板卡
BOARD="${BOARD_IP:-192.168.5.13}"
BOARD_USER="${BOARD_USER:-HwHiAiUser}"
BOARD_DIR="/home/$BOARD_USER/ICT"

echo "推送到 $BOARD_USER@$BOARD:$BOARD_DIR ..."

rsync -avz --relative \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.om' \
  --exclude='*.onnx' \
  --exclude='*.pt' \
  --exclude='remote_sync/' \
  --exclude='.git/' \
  --exclude='models/' \
  --exclude='kernel_meta/' \
  --exclude='.wheelhouse*/' \
  --exclude='*.7z' \
  --exclude='*.tar.gz' \
  ./ "$BOARD_USER@$BOARD:$BOARD_DIR/"

echo ""
echo "推送完成！"
