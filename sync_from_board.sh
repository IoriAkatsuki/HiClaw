#!/bin/bash
# 从板卡同步 git 仓库到本地

echo "正在从板卡同步 ICT 项目..."

rsync -avz --progress \
  --exclude='*.log' \
  --exclude='nohup.out' \
  --exclude='*.om' \
  --exclude='*.onnx' \
  --exclude='*.pt' \
  --exclude='*.pth' \
  --exclude='runs/' \
  --exclude='ElectroCom61*/' \
  --exclude='samples_source/' \
  --exclude='qwen25_fastllm/' \
  --exclude='qwen_onnx/' \
  --exclude='build/' \
  --exclude='__pycache__/' \
  --exclude='webui_http/frame.jpg' \
  --exclude='webui_http/state.json' \
  --exclude='*.pyc' \
  HwHiAiUser@ict.local:/home/HwHiAiUser/ICT/ /home/oasis/Documents/ICT/remote_sync/

echo ""
echo "同步完成！"
echo "本地仓库: /home/oasis/Documents/ICT/remote_sync/"

cd /home/oasis/Documents/ICT/remote_sync/
echo ""
echo "Git 状态:"
git log --oneline -5
echo ""
git status -s | head -10
