#!/bin/bash
set -e

# 一键启动 YOLO 实时检测 + WebUI

APP_HOME="$HOME/ICT"
WEB_DIR="$APP_HOME/webui"
YOLO_DIR="$APP_HOME/YoloV3Infer"
CAM="/dev/video4"

# Ascend / mxVision 环境（复制到用户目录，避免权限问题）
MX_SDK_HOME="$HOME/mxVision-6.0.0.SPC2"
export MX_SDK_HOME
export LD_LIBRARY_PATH="$MX_SDK_HOME/opensource/lib:$MX_SDK_HOME/lib:$MX_SDK_HOME/lib/modelpostprocessors:$LD_LIBRARY_PATH"

# 简单的 sudo 封装（用于 v4l2-ctl、pkill root 进程）
sudo_run() {
  echo ict | sudo -S "$@"
}

# 调整摄像头曝光/亮度/帧率，防止画面过暗且锁定 15fps
sudo_run v4l2-ctl -d "$CAM" \
  --set-ctrl brightness=20,contrast=60,saturation=80,gain=90,exposure_auto=1,exposure_absolute=400 \
  --set-parm=15 \
  >/dev/null 2>&1 || true

# 结束旧的推理与 HTTP 进程
sudo_run pkill -f mxbaseV2_sample 2>/dev/null || true
pkill -f "python3 -m http.server 8000" 2>/dev/null || true

# 启动 YOLO 常驻推理
cd "$YOLO_DIR"
nohup ./mxbaseV2_sample "$CAM" > "$WEB_DIR/live.log" 2>&1 &

# 启动 Web 服务
cd "$WEB_DIR"
nohup python3 -m http.server 8000 > http.log 2>&1 &

echo "YOLO + WebUI 已启动，在浏览器访问: http://<板卡IP>:8000/"
