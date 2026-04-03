#!/bin/bash
# 一键演示启动脚本 — 拉起 WebUI + Chat + 统一检测监控
# 用法: ./start_demo.sh

set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

# Ascend 环境
if [ -f ~/miniconda3/Ascend/cann/set_env.sh ]; then
    source ~/miniconda3/Ascend/cann/set_env.sh 2>/dev/null
elif [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null
fi

LOG_DIR="$HOME/ICT/logs"
mkdir -p "$LOG_DIR"

PYTHON_BIN="${ICT_PYTHON_BIN:-$(command -v python3)}"
PYTHON_SITE_PACKAGES="$("$PYTHON_BIN" -c 'import site; paths = [site.getusersitepackages(), *site.getsitepackages()]; seen = []; [seen.append(p) for p in paths if p and p not in seen]; print(":".join(seen))' 2>/dev/null || true)"
if [ -n "$PYTHON_SITE_PACKAGES" ]; then
    export PYTHONPATH="$PYTHON_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$HOME/ICT/pybind_venc/build${PYTHONPATH:+:$PYTHONPATH}"

# 统一检测监控的模型配置
BOARD_MODEL_DIR="/home/HwHiAiUser/ICT/d435_project/projects/yolo26_galvo/models"
DEFAULT_YOLO_MODEL="$HOME/ICT/models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om"
CONTROL_PLANE="$HOME/ICT/edge/unified_app/control_plane.py"

find_preferred_yolo_model() {
    local candidate=""
    if [ -d "$BOARD_MODEL_DIR" ]; then
        for pattern in "*yolom*.om" "*yolo26m*.om" "*yolom*.mindir" "*yolo26m*.mindir" "*yolo26n*.om" "*yolo26n*.mindir"; do
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
    eval "$("$PYTHON_BIN" "$CONTROL_PLANE" shell-env "$HOME/ICT")"
fi

YOLO_MODEL="${YOLO_MODEL:-${CONTROL_YOLO_MODEL:-$(find_preferred_yolo_model)}}"
DATA_YAML="${DATA_YAML:-${CONTROL_DATA_YAML:-config/yolo26_6cls.yaml}}"
ENABLE_HAND_DETECTION="${ENABLE_HAND_DETECTION:-${CONTROL_ENABLE_HAND_DETECTION:-0}}"
ENABLE_DISTANCE_DETECTION="${ENABLE_DISTANCE_DETECTION:-${CONTROL_ENABLE_DISTANCE_DETECTION:-0}}"
ENABLE_VIDEO_STREAM="${ENABLE_VIDEO_STREAM:-${CONTROL_ENABLE_VIDEO_STREAM:-1}}"
DANGER_DISTANCE="${DANGER_DISTANCE:-${CONTROL_DANGER_DISTANCE:-300}}"
CONF_THRES="${CONF_THRES:-${CONTROL_CONF_THRES:-0.55}}"
CAMERA_SERIAL="${CAMERA_SERIAL:-${CONTROL_CAMERA_SERIAL:-}}"
ENABLE_LASER="${ENABLE_LASER:-${CONTROL_ENABLE_LASER:-0}}"
LASER_SERIAL="${LASER_SERIAL:-${CONTROL_LASER_SERIAL:-/dev/ttyUSB0}}"
LASER_BAUDRATE="${LASER_BAUDRATE:-${CONTROL_LASER_BAUDRATE:-115200}}"
LASER_CALIBRATION="${LASER_CALIBRATION:-${CONTROL_LASER_CALIBRATION:-$HOME/ICT/edge/laser_galvo/galvo_calibration.yaml}}"

echo "=========================================="
echo "  Laser-AR 演示系统 一键启动"
echo "=========================================="

# 1. 清理残留进程
echo ""
echo "[1/4] 清理残留进程..."
pkill -f webui_server.py 2>/dev/null && echo "  stopped: webui_server" || true
pkill -f llama-server 2>/dev/null && echo "  stopped: llama-server" || true
pkill -f unified_monitor 2>/dev/null && echo "  stopped: unified_monitor" || true
sleep 1

# 2. 启动 WebUI（含 Chat 后端）
echo ""
echo "[2/4] 启动 WebUI Server..."
nohup "$PYTHON_BIN" edge/unified_app/webui_server.py \
    > "$LOG_DIR/webui_demo.log" 2>&1 &
WEBUI_PID=$!
echo "  PID: $WEBUI_PID"

# 等待 WebUI 就绪
for i in $(seq 1 15); do
    sleep 2
    if curl -s http://127.0.0.1:8002/api/chat/status 2>/dev/null | grep -q '"backends"'; then
        echo "  WebUI 就绪"
        break
    fi
    [ "$i" -eq 15 ] && echo "  警告: WebUI 启动超时"
done

# 3. 启动统一检测监控（后台）
echo ""
echo "[3/4] 启动统一检测监控..."
echo "  YOLO 模型: $YOLO_MODEL"
echo "  手部模型: MediaPipe Hands"
echo "  配置文件: $DATA_YAML"
echo "  危险距离: ${DANGER_DISTANCE}mm"

if [ ! -f "$YOLO_MODEL" ]; then
    echo "  ✗ 未找到 YOLO 模型: $YOLO_MODEL"
    echo "    跳过检测监控（Chat 仍可用）"
else
    CMD=(
        "$PYTHON_BIN" edge/unified_app/unified_monitor_mp.py
        --yolo-model "$YOLO_MODEL"
        --data-yaml "$DATA_YAML"
        --danger-distance "$DANGER_DISTANCE"
        --conf-thres "$CONF_THRES"
    )
    [ -n "$CAMERA_SERIAL" ] && CMD+=(--camera-serial "$CAMERA_SERIAL")
    [ "$ENABLE_HAND_DETECTION" != "1" ] && CMD+=(--disable-hand)
    [ "$ENABLE_DISTANCE_DETECTION" != "1" ] && CMD+=(--disable-distance)
    [ "$ENABLE_VIDEO_STREAM" != "1" ] && CMD+=(--disable-video-stream)
    if [ "$ENABLE_LASER" = "1" ]; then
        CMD+=(--enable-laser --laser-serial "$LASER_SERIAL" --laser-baudrate "$LASER_BAUDRATE")
        [ -n "$LASER_CALIBRATION" ] && CMD+=(--laser-calibration "$LASER_CALIBRATION")
    fi

    nohup "${CMD[@]}" > "$LOG_DIR/unified_demo.log" 2>&1 &
    MONITOR_PID=$!
    echo "  PID: $MONITOR_PID"
    sleep 3
    if kill -0 "$MONITOR_PID" 2>/dev/null; then
        echo "  ✓ 检测监控已启动"
    else
        echo "  ✗ 检测监控启动失败，查看日志: $LOG_DIR/unified_demo.log"
    fi
fi

# 4. 状态汇总
echo ""
echo "[4/4] 服务状态"
echo "=========================================="

if curl -s http://127.0.0.1:8002/api/chat/status > /dev/null 2>&1; then
    echo "  ✓ WebUI:    http://$(hostname).local:8002"
else
    echo "  ✗ WebUI 未响应"
fi

BACKEND=$(curl -s http://127.0.0.1:8002/api/chat/status 2>/dev/null | python3 -c "
import sys,json
try:
    d=json.load(sys.stdin)
    print(d.get('backend') or '未初始化（首次对话时自动加载）')
except: print('未知')
" 2>/dev/null)
echo "  ✓ Chat:     $BACKEND"

if pgrep -f unified_monitor > /dev/null 2>&1; then
    echo "  ✓ 检测监控: 运行中"
else
    echo "  - 检测监控: 未运行"
fi

echo ""
echo "=========================================="
echo "  日志: $LOG_DIR/webui_demo.log"
echo "        $LOG_DIR/unified_demo.log"
echo "  停止: pkill -f webui_server.py; pkill -f unified_monitor"
echo "=========================================="
