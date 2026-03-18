#!/usr/bin/env bash
# 启动统一监控系统（带激光振镜标记）
#
# 使用方法:
#   1. 不带激光: ./start_unified_with_laser.sh
#   2. 带激光:   ./start_unified_with_laser.sh --enable-laser

set -euo pipefail

# 基础路径（相对脚本自动解析，避免依赖固定 ~/ICT）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ICT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 模型和配置（允许通过环境变量覆盖）
YOLO_MODEL="${YOLO_MODEL:-$ICT_DIR/models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om}"
POSE_MODEL="${POSE_MODEL:-$ICT_DIR/yolov8n_pose_aipp.om}"
DATA_YAML="${DATA_YAML:-$ICT_DIR/config/yolo26_6cls.yaml}"

# 参数（允许通过环境变量覆盖）
DANGER_DISTANCE="${DANGER_DISTANCE:-300}"
CONF_THRES="${CONF_THRES:-0.55}"
YOLO_DEVICE="${YOLO_DEVICE:-cpu}"
LASER_SERIAL="${LASER_SERIAL:-/dev/ttyUSB0}"
LASER_BAUDRATE="${LASER_BAUDRATE:-115200}"
LASER_CALIBRATION="${LASER_CALIBRATION:-$ICT_DIR/edge/laser_galvo/galvo_calibration.yaml}"

echo "=========================================="
echo "统一检测监控 - 物体 + 手部 + 激光标记"
echo "=========================================="
echo ""

# 检查基础文件
if [ ! -f "$YOLO_MODEL" ]; then
    echo "错误: YOLO模型不存在: $YOLO_MODEL"
    echo "提示: 先执行 $ICT_DIR/tools/export_latest_yolo26_to_om.sh 生成最新 yolo26 的 OM 文件"
    exit 1
fi

if [ ! -f "$DATA_YAML" ]; then
    echo "错误: 数据配置不存在: $DATA_YAML"
    exit 1
fi

# 构建命令
CMD=(
    /usr/bin/python3 "$SCRIPT_DIR/unified_monitor_mp.py"
    --yolo-model "$YOLO_MODEL"
    --yolo-device "$YOLO_DEVICE"
    --data-yaml "$DATA_YAML"
    --danger-distance "$DANGER_DISTANCE"
    --conf-thres "$CONF_THRES"
)

if [ -f "$POSE_MODEL" ]; then
    CMD+=(--pose-model "$POSE_MODEL")
    echo "手部模型: 使用 YOLO pose ($POSE_MODEL)"
else
    echo "手部模型: 未找到 pose OM，回退到 MediaPipe ($POSE_MODEL)"
fi

# 检查是否启用激光
if [[ " $* " == *" --enable-laser "* ]]; then
    echo "激光振镜: 启用"
    echo "  串口: $LASER_SERIAL"
    echo "  波特率: $LASER_BAUDRATE"

    # 检查串口设备
    if [ ! -e "$LASER_SERIAL" ]; then
        echo "警告: 串口设备不存在: $LASER_SERIAL"
        echo "      请检查STM32是否已连接"
        read -r -p "是否继续运行（不带激光）？[y/N] " REPLY
        if [[ ! "$REPLY" =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        CMD+=(
            --enable-laser
            --laser-serial "$LASER_SERIAL"
            --laser-baudrate "$LASER_BAUDRATE"
        )

        if [ -f "$LASER_CALIBRATION" ]; then
            CMD+=(--laser-calibration "$LASER_CALIBRATION")
            echo "  标定文件: $LASER_CALIBRATION"
        else
            echo "  提示: 未找到标定文件，继续运行（使用默认映射）"
        fi
    fi
else
    echo "激光振镜: 禁用"
    echo "  提示: 使用 --enable-laser 参数启用激光标记"
fi

echo ""
echo "启动监控..."
echo ""

exec "${CMD[@]}"
