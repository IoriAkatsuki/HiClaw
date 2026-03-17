#!/bin/bash
# 一键自动校准脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "  激光振镜一键自动校准"
echo "=========================================="
echo ""

# 检查Python依赖
echo "检查依赖..."
python3 -c "import cv2, numpy, yaml, serial" 2>/dev/null || {
    echo "✗ 缺少Python依赖"
    echo ""
    echo "安装依赖:"
    echo "  pip3 install opencv-python numpy pyyaml pyserial"
    exit 1
}
echo "✓ 依赖检查通过"
echo ""

# 检查串口
echo "检查串口设备..."
if ls /dev/ttyUSB* >/dev/null 2>&1 || ls /dev/ttyACM* >/dev/null 2>&1; then
    SERIAL_DEV=$(ls /dev/ttyUSB* /dev/ttyACM* 2>/dev/null | head -1)
    echo "✓ 检测到: $SERIAL_DEV"
else
    echo "✗ 未检测到串口设备"
    echo ""
    echo "请检查:"
    echo "  1. STM32 USB线是否已连接"
    echo "  2. 驱动是否已加载"
    echo ""
    exit 1
fi
echo ""

# 检查摄像头
echo "检查摄像头..."
if [ -e /dev/video0 ]; then
    echo "✓ 检测到: /dev/video0"
else
    echo "✗ 未检测到摄像头"
    exit 1
fi
echo ""

# 运行自动校准
echo "启动自动校准程序..."
echo "=========================================="
echo ""

cd "$SCRIPT_DIR"
python3 auto_calibrate.py

echo ""
echo "=========================================="
echo "校准完成"
echo "=========================================="
