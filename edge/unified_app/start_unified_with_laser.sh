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
YOLO_MODEL="${YOLO_MODEL:-$ICT_DIR/runs/detect/train_electro61/weights/yolov8_electro61_aipp.om}"
DATA_YAML="${DATA_YAML:-$ICT_DIR/config/electro61.yaml}"

# 参数（允许通过环境变量覆盖）
DANGER_DISTANCE="${DANGER_DISTANCE:-300}"
CONF_THRES="${CONF_THRES:-0.55}"
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
    exit 1
fi

if [ ! -f "$DATA_YAML" ]; then
    echo "错误: 数据配置不存在: $DATA_YAML"
    exit 1
fi

# 构建命令
CMD=(
    python3 "$SCRIPT_DIR/unified_monitor.py"
    --yolo-model "$YOLO_MODEL"
    --data-yaml "$DATA_YAML"
    --danger-distance "$DANGER_DISTANCE"
    --conf-thres "$CONF_THRES"
)

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
