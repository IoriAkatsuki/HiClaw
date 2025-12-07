#!/bin/bash
# Hand Safety Monitor - Route B 启动脚本

cd ~/ICT

echo "=========================================="
echo "Hand Safety Monitor - Route B"
echo "=========================================="

# 1. 启动 WebUI 服务器 (后台)
echo "[1/2] 启动 WebUI 服务器 (端口 8001)..."
python3 edge/route_b_app/webui_server.py &
WEBUI_PID=$!
sleep 1

if ps -p $WEBUI_PID > /dev/null; then
    echo "✓ WebUI 服务器已启动 (PID: $WEBUI_PID)"
    echo "  访问: http://ict.local:8001"
else
    echo "✗ WebUI 服务器启动失败"
    exit 1
fi

# 2. 启动手部安全监控
echo ""
echo "[2/2] 启动手部安全监控..."
echo "=========================================="

source /usr/local/Ascend/ascend-toolkit/set_env.sh

python3 edge/route_b_app/hand_safety_monitor.py \
    --model yolov8n_pose_aipp.om \
    --danger-distance 150 \
    --conf-thres 0.5

# 清理
kill $WEBUI_PID 2>/dev/null
echo ""
echo "✓ Route B 已停止"
