#!/bin/bash
# 一键启动：杀死所有占用进程 → 启动 WebUI + 激光检测
# 用法: bash start_laser_webui.sh [串口设备]
#   例: bash start_laser_webui.sh /dev/ttyUSB0

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 强制使用系统 Python（~/bin/python3 缺少 pip 和系统 dist-packages）
export ICT_PYTHON_BIN="/usr/bin/python3"

SERIAL="${1:-}"

echo "══════════════════════════════════════════"
echo " 一键启动 Laser-AR 检测 + WebUI"
echo "══════════════════════════════════════════"

# ── 1. 杀死所有占用进程 ──
echo ""
echo "[1/4] 清理旧进程..."
killall -9 python3 2>/dev/null || true
pkill -9 -f start_unified 2>/dev/null || true
sleep 2
echo "  ✓ 已清理"

# ── 2. 自动检测串口 ──
echo ""
echo "[2/4] 检测振镜串口..."
if [ -z "$SERIAL" ]; then
    for dev in /dev/ttyUSB0 /dev/ttyUSB1 /dev/ttyACM0; do
        if [ -e "$dev" ]; then
            SERIAL="$dev"
            break
        fi
    done
fi
if [ -z "$SERIAL" ]; then
    echo "  ⚠ 未找到串口设备，激光将禁用"
else
    echo "  ✓ 使用串口: $SERIAL"
    # 更新 runtime config 中的串口
    python3 -c "
import json, pathlib
p = pathlib.Path('runtime/unified_control.json')
if p.exists():
    d = json.loads(p.read_text())
    d['laser_serial'] = '$SERIAL'
    d['enable_laser'] = True
    p.write_text(json.dumps(d, indent=4))
" 2>/dev/null || true
fi

# ── 3. 加载 Ascend 环境 ──
echo ""
echo "[3/4] 加载 Ascend 环境..."
if [ -f "$HOME/miniconda3/Ascend/cann/set_env.sh" ]; then
    source "$HOME/miniconda3/Ascend/cann/set_env.sh"
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    echo "  ✗ 未找到 Ascend 环境"
    exit 1
fi
export PYTHONPATH="$HOME/ICT/pybind_venc/build${PYTHONPATH:+:$PYTHONPATH}"
echo "  ✓ Ascend 环境已加载"

# ── 4. 启动服务 ──
echo ""
echo "[4/4] 启动服务..."

# 强制设置激光相关环境变量
if [ -n "$SERIAL" ]; then
    export ENABLE_LASER=1
    export LASER_SERIAL="$SERIAL"
fi

# 使用 MediaPipe 做手部检测（不传 pose model 给 unified_monitor）
export POSE_MODEL="none"

mkdir -p logs
bash ./start_unified.sh &
UNIFIED_PID=$!

sleep 5
echo ""
echo "══════════════════════════════════════════"
echo " ✓ 启动完成"
echo "   WebUI:  http://$(hostname -I | awk '{print $1}'):8002"
echo "   激光:   ${SERIAL:-禁用}"
echo "   PID:    $UNIFIED_PID"
echo "══════════════════════════════════════════"

wait $UNIFIED_PID
