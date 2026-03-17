#!/bin/bash
# 激光振镜系统快速启动脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "激光振镜物体标注系统 - 快速启动"
echo "========================================"

# 检查Python依赖
echo "检查依赖..."
python3 -c "import cv2, numpy, serial, yaml" || {
    echo "缺少依赖，正在安装..."
    pip3 install opencv-python numpy pyserial pyyaml
}

# 菜单
echo ""
echo "请选择操作:"
echo "  1) 生成标定板"
echo "  2) 执行自动标定"
echo "  3) 测试标定精度"
echo "  4) 运行完整系统（不启用激光）"
echo "  5) 运行完整系统（启用激光）"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        echo "生成标定板..."
        python3 generate_calibration_target.py
        echo ""
        echo "标定板已生成，请打印以下文件:"
        echo "  /home/oasis/Documents/ICT/calibration_data/laser_calibration_board_A4.png"
        ;;

    2)
        read -p "串口设备路径 [/dev/ttyUSB0]: " serial_port
        serial_port=${serial_port:-/dev/ttyUSB0}

        read -p "波特率 [115200]: " baudrate
        baudrate=${baudrate:-115200}

        echo "开始标定..."
        python3 calibrate_galvo.py \
            --serial-port "$serial_port" \
            --baudrate "$baudrate" \
            --output galvo_calibration.yaml
        ;;

    3)
        read -p "串口设备路径 [/dev/ttyUSB0]: " serial_port
        serial_port=${serial_port:-/dev/ttyUSB0}

        echo "启动测试..."
        python3 calibrate_galvo.py \
            --serial-port "$serial_port" \
            --test \
            --load galvo_calibration.yaml
        ;;

    4)
        cd ../unified_app
        python3 unified_monitor_with_laser.py \
            --yolo-model ../models/yolov8n_electro61.om \
            --data-yaml ../data/electro61.yaml \
            --danger-distance 300 \
            --conf-thres 0.55
        ;;

    5)
        if [ ! -f "galvo_calibration.yaml" ]; then
            echo "错误: 未找到标定文件，请先执行标定 (选项2)"
            exit 1
        fi

        read -p "串口设备路径 [/dev/ttyUSB0]: " serial_port
        serial_port=${serial_port:-/dev/ttyUSB0}

        read -p "最小标注置信度 [0.7]: " min_score
        min_score=${min_score:-0.7}

        read -p "目标类别 (空格分隔，留空=全部): " target_classes

        cd ../unified_app
        if [ -z "$target_classes" ]; then
            python3 unified_monitor_with_laser.py \
                --yolo-model ../models/yolov8n_electro61.om \
                --data-yaml ../data/electro61.yaml \
                --danger-distance 300 \
                --conf-thres 0.55 \
                --enable-laser \
                --laser-serial "$serial_port" \
                --laser-calibration ../laser_galvo/galvo_calibration.yaml \
                --laser-min-score "$min_score"
        else
            python3 unified_monitor_with_laser.py \
                --yolo-model ../models/yolov8n_electro61.om \
                --data-yaml ../data/electro61.yaml \
                --danger-distance 300 \
                --conf-thres 0.55 \
                --enable-laser \
                --laser-serial "$serial_port" \
                --laser-calibration ../laser_galvo/galvo_calibration.yaml \
                --laser-min-score "$min_score" \
                --laser-target-classes $target_classes
        fi
        ;;

    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "操作完成！"
