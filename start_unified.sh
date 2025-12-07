#!/bin/bash
# Unified Detection Monitor Startup Script
# 统一检测监控启动脚本 - 物体检测 + 手部安全

set -e

cd ~/ICT

echo "======================================================================"
echo "统一检测监控 - 物体 + 手部"
echo "======================================================================"

# 1. 加载 Ascend 环境
echo "[1/3] 加载 Ascend 环境..."
source /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "✓ Ascend 环境已加载"

# 2. 启动 WebUI 服务器 (后台)
echo ""
echo "[2/3] 启动 WebUI 服务器 (端口 8002)..."
pkill -f "webui_server.py" 2>/dev/null || true
nohup python3 edge/unified_app/webui_server.py > logs/webui_unified.log 2>&1 &
sleep 2
echo "✓ WebUI 服务器已启动: http://ict.local:8002"

# 3. 启动统一监控
echo ""
echo "[3/3] 启动统一检测监控..."
echo "  YOLO 模型: runs/detect/train_electro61/weights/yolov8_electro61_aipp.om"
echo "  配置文件: config/electro61.yaml"
echo "  危险距离: 300mm (30cm)"
echo ""
python3 edge/unified_app/unified_monitor.py \
    --yolo-model runs/detect/train_electro61/weights/yolov8_electro61_aipp.om \
    --data-yaml config/electro61.yaml \
    --danger-distance 300 \
    --conf-thres 0.55
