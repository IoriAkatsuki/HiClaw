#!/bin/bash
# Unified Detection Monitor Startup Script
# 统一检测监控启动脚本 - 物体检测 + 手部安全

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICT_DIR="$SCRIPT_DIR"
BOARD_MODEL_DIR="/home/HwHiAiUser/ICT/d435_project/projects/yolo26_galvo/models"
DEFAULT_YOLO_MODEL="$ICT_DIR/models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om"
CONTROL_PLANE="$ICT_DIR/edge/unified_app/control_plane.py"

find_preferred_yolo_model() {
    local candidate=""
    if [ -d "$BOARD_MODEL_DIR" ]; then
        for pattern in "*yolo26n*.om" "*yolo26n*.mindir" "*yolom*.om" "*yolo26m*.om" "*yolom*.mindir" "*yolo26m*.mindir"; do
            candidate=$(find "$BOARD_MODEL_DIR" -maxdepth 2 -type f -iname "$pattern" 2>/dev/null | sort | head -n 1)
            if [ -n "$candidate" ]; then
                printf '%s\n' "$candidate"
                return 0
            fi
        done
    fi
    printf '%s\n' "$DEFAULT_YOLO_MODEL"
}

if [ -f "$CONTROL_PLANE" ]; then
    # 通过 control_plane.py shell-env 注入统一控制配置。
    eval "$(python3 "$ICT_DIR/edge/unified_app/control_plane.py" shell-env "$ICT_DIR")"
fi

YOLO_MODEL="${YOLO_MODEL:-${CONTROL_YOLO_MODEL:-$(find_preferred_yolo_model)}}"
POSE_MODEL="${POSE_MODEL:-${CONTROL_POSE_MODEL:-$HOME/ICT/yolov8n_pose_aipp.om}}"
DATA_YAML="${DATA_YAML:-${CONTROL_DATA_YAML:-config/yolo26_6cls.yaml}}"
DANGER_DISTANCE="${DANGER_DISTANCE:-${CONTROL_DANGER_DISTANCE:-300}}"
CONF_THRES="${CONF_THRES:-${CONTROL_CONF_THRES:-0.55}}"
CAMERA_SERIAL="${CAMERA_SERIAL:-${CONTROL_CAMERA_SERIAL:-}}"
ENABLE_LASER="${ENABLE_LASER:-${CONTROL_ENABLE_LASER:-0}}"
LASER_SERIAL="${LASER_SERIAL:-${CONTROL_LASER_SERIAL:-/dev/ttyUSB0}}"
LASER_BAUDRATE="${LASER_BAUDRATE:-${CONTROL_LASER_BAUDRATE:-115200}}"
LASER_CALIBRATION="${LASER_CALIBRATION:-${CONTROL_LASER_CALIBRATION:-$ICT_DIR/edge/laser_galvo/galvo_calibration.yaml}}"

cd "$ICT_DIR"

echo "======================================================================"
echo "统一检测监控 - 物体 + 手部"
echo "======================================================================"

# 1. 加载 Ascend 环境
echo "[1/3] 加载 Ascend 环境..."
if [ -f "$HOME/miniconda3/Ascend/cann/set_env.sh" ]; then
    # 优先使用板端当前已验证的 CANN 8.5 环境
    source "$HOME/miniconda3/Ascend/cann/set_env.sh"
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    echo "✗ 未找到 Ascend 环境脚本"
    exit 1
fi
export PYTHONPATH="$HOME/ICT/pybind_venc/build${PYTHONPATH:+:$PYTHONPATH}"
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
echo "  YOLO 模型: $YOLO_MODEL"
echo "  Pose 模型: $POSE_MODEL"
echo "  配置文件: $DATA_YAML"
echo "  危险距离: ${DANGER_DISTANCE}mm"
echo ""
if [ ! -f "$YOLO_MODEL" ]; then
    echo "✗ 未找到 YOLO 模型: $YOLO_MODEL"
    echo "  请先运行: bash tools/export_latest_yolo26_to_om.sh"
    exit 1
fi

CMD=(
    /usr/bin/python3 edge/unified_app/unified_monitor_mp.py
    --yolo-model "$YOLO_MODEL"
    --data-yaml "$DATA_YAML"
    --danger-distance "$DANGER_DISTANCE"
    --conf-thres "$CONF_THRES"
)

if [ -n "$CAMERA_SERIAL" ]; then
    CMD+=(--camera-serial "$CAMERA_SERIAL")
fi

if [ "$ENABLE_LASER" = "1" ]; then
    CMD+=(--enable-laser --laser-serial "$LASER_SERIAL" --laser-baudrate "$LASER_BAUDRATE")
    if [ -n "$LASER_CALIBRATION" ]; then
        CMD+=(--laser-calibration "$LASER_CALIBRATION")
    fi
fi

if [ -f "$POSE_MODEL" ]; then
    CMD+=(--pose-model "$POSE_MODEL")
    echo "  使用 YOLO pose: $POSE_MODEL"
else
    echo "  未找到 pose OM，回退到 MediaPipe: $POSE_MODEL"
fi

"${CMD[@]}"
