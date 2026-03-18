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
echo "  YOLO 模型: models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om"
echo "  Pose 模型: ${POSE_MODEL:-yolov8n_pose_aipp.om}"
echo "  配置文件: config/yolo26_6cls.yaml"
echo "  危险距离: 300mm (30cm)"
echo ""
if [ ! -f models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om ]; then
    echo "✗ 未找到正式 OM 模型: models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om"
    echo "  请先运行: bash tools/export_latest_yolo26_to_om.sh"
    exit 1
fi

POSE_MODEL="${POSE_MODEL:-$HOME/ICT/yolov8n_pose_aipp.om}"
CMD=(
    /usr/bin/python3 edge/unified_app/unified_monitor_mp.py
    --yolo-model models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om
    --data-yaml config/yolo26_6cls.yaml
    --danger-distance 300
    --conf-thres 0.55
)

if [ -f "$POSE_MODEL" ]; then
    CMD+=(--pose-model "$POSE_MODEL")
    echo "  使用 YOLO pose: $POSE_MODEL"
else
    echo "  未找到 pose OM，回退到 MediaPipe: $POSE_MODEL"
fi

"${CMD[@]}"
