#!/bin/bash
# 一键启动智能检测系统
# 运行后可在浏览器访问 http://ict.local:8002

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量
ICT_DIR="$HOME/ICT"
LOG_DIR="$ICT_DIR/logs"
PID_DIR="$ICT_DIR/pids"
WEBUI_PORT=8002
WEBUI_DIR="$HOME/ICT/webui_http_unified"

# 默认参数
YOLO_MODEL="$ICT_DIR/models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om"
POSE_MODEL="${POSE_MODEL:-$ICT_DIR/yolov8n_pose_aipp.om}"
DATA_YAML="$ICT_DIR/config/yolo26_6cls.yaml"
DANGER_DISTANCE=300
CONF_THRES=0.6
YOLO_DEVICE="cpu"

# 创建必要目录
mkdir -p "$LOG_DIR" "$PID_DIR" "$WEBUI_DIR"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}       智能激光视觉分拣系统 - 一键启动${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# 函数：检查进程是否运行
check_process() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # 运行中
        else
            rm -f "$pid_file"
            return 1  # 未运行
        fi
    fi
    return 1  # 文件不存在
}

# 函数：停止进程
stop_process() {
    local name=$1
    local pid_file=$2
    
    if check_process "$pid_file"; then
        local pid=$(cat "$pid_file")
        echo -e "${YELLOW}停止 $name (PID: $pid)...${NC}"
        kill "$pid" 2>/dev/null || true
        sleep 2
        
        # 如果进程还在运行，强制杀死
        if ps -p "$pid" > /dev/null 2>&1; then
            echo -e "${YELLOW}强制停止 $name...${NC}"
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
        echo -e "${GREEN}✓ $name 已停止${NC}"
    fi
}

# 函数：启动WebUI服务器
start_webui() {
    local pid_file="$PID_DIR/webui.pid"
    
    if check_process "$pid_file"; then
        echo -e "${GREEN}✓ WebUI服务器已在运行${NC}"
        return 0
    fi
    
    echo -e "${BLUE}启动 WebUI 服务器...${NC}"

    # 启动仓库内正式 WebUI 服务器
    cd "$ICT_DIR/edge/unified_app"
    nohup /usr/bin/python3 webui_server.py > "$LOG_DIR/webui.log" 2>&1 &
    echo $! > "$pid_file"
    
    sleep 2
    
    if check_process "$pid_file"; then
        echo -e "${GREEN}✓ WebUI服务器已启动 (PID: $(cat $pid_file))${NC}"
        echo -e "${BLUE}  访问地址: http://ict.local:$WEBUI_PORT${NC}"
    else
        echo -e "${RED}✗ WebUI服务器启动失败${NC}"
        return 1
    fi
}

# 函数：启动检测程序
start_detection() {
    local pid_file="$PID_DIR/detection.pid"
    
    if check_process "$pid_file"; then
        echo -e "${GREEN}✓ 检测程序已在运行${NC}"
        return 0
    fi
    
    echo -e "${BLUE}启动检测程序...${NC}"
    
    # 加载Ascend环境
    if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
    
    # 检查模型文件
    if [ ! -f "$YOLO_MODEL" ]; then
        echo -e "${RED}✗ 模型文件不存在: $YOLO_MODEL${NC}"
        echo -e "${YELLOW}  请先运行: $ICT_DIR/tools/export_latest_yolo26_to_om.sh${NC}"
        return 1
    fi
    
    # 启动检测程序
    cd "$ICT_DIR/edge/unified_app"
    export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
    if [ -f "$POSE_MODEL" ]; then
        nohup /usr/bin/python3 unified_monitor_mp.py \
            --yolo-model "$YOLO_MODEL" \
            --yolo-device "$YOLO_DEVICE" \
            --pose-model "$POSE_MODEL" \
            --data-yaml "$DATA_YAML" \
            --danger-distance "$DANGER_DISTANCE" \
            --conf-thres "$CONF_THRES" \
            > "$LOG_DIR/detection.log" 2>&1 &
    else
        nohup /usr/bin/python3 unified_monitor_mp.py \
            --yolo-model "$YOLO_MODEL" \
            --yolo-device "$YOLO_DEVICE" \
            --data-yaml "$DATA_YAML" \
            --danger-distance "$DANGER_DISTANCE" \
            --conf-thres "$CONF_THRES" \
            > "$LOG_DIR/detection.log" 2>&1 &
    fi
    echo $! > "$pid_file"
    
    sleep 3
    
    if check_process "$pid_file"; then
        echo -e "${GREEN}✓ 检测程序已启动 (PID: $(cat $pid_file))${NC}"
    else
        echo -e "${RED}✗ 检测程序启动失败${NC}"
        echo -e "${YELLOW}  查看日志: tail -f $LOG_DIR/detection.log${NC}"
        return 1
    fi
}

# 函数：停止所有服务
stop_all() {
    echo -e "${BLUE}停止所有服务...${NC}"
    stop_process "WebUI服务器" "$PID_DIR/webui.pid"
    stop_process "检测程序" "$PID_DIR/detection.pid"
    echo -e "${GREEN}✓ 所有服务已停止${NC}"
}

# 函数：显示状态
show_status() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}系统状态${NC}"
    echo -e "${BLUE}============================================================${NC}"
    
    # WebUI状态
    if check_process "$PID_DIR/webui.pid"; then
        echo -e "${GREEN}✓ WebUI服务器: 运行中 (PID: $(cat $PID_DIR/webui.pid))${NC}"
        echo -e "  访问地址: ${BLUE}http://ict.local:$WEBUI_PORT${NC}"
    else
        echo -e "${RED}✗ WebUI服务器: 未运行${NC}"
    fi
    
    # 检测程序状态
    if check_process "$PID_DIR/detection.pid"; then
        echo -e "${GREEN}✓ 检测程序: 运行中 (PID: $(cat $PID_DIR/detection.pid))${NC}"
    else
        echo -e "${RED}✗ 检测程序: 未运行${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}日志文件:${NC}"
    echo -e "  WebUI: $LOG_DIR/webui.log"
    echo -e "  检测: $LOG_DIR/detection.log"
    echo ""
}

# 函数：显示日志
show_logs() {
    local log_type=$1
    
    case $log_type in
        webui)
            tail -f "$LOG_DIR/webui.log"
            ;;
        detection)
            tail -f "$LOG_DIR/detection.log"
            ;;
        all)
            tail -f "$LOG_DIR/webui.log" "$LOG_DIR/detection.log"
            ;;
        *)
            echo "用法: $0 logs [webui|detection|all]"
            ;;
    esac
}

# 主程序
case "${1:-start}" in
    start)
        echo -e "${BLUE}启动系统...${NC}"
        start_webui
        start_detection
        echo ""
        show_status
        echo -e "${GREEN}============================================================${NC}"
        echo -e "${GREEN}系统启动完成！${NC}"
        echo -e "${GREEN}============================================================${NC}"
        echo ""
        echo -e "在浏览器中访问: ${BLUE}http://ict.local:$WEBUI_PORT${NC}"
        echo -e "或使用IP: ${BLUE}http://$(hostname -I | awk '{print $1}'):$WEBUI_PORT${NC}"
        echo ""
        echo -e "命令:"
        echo -e "  $0 status   - 查看状态"
        echo -e "  $0 stop     - 停止服务"
        echo -e "  $0 restart  - 重启服务"
        echo -e "  $0 logs     - 查看日志"
        echo ""
        ;;
    
    stop)
        stop_all
        ;;
    
    restart)
        echo -e "${BLUE}重启系统...${NC}"
        stop_all
        sleep 2
        start_webui
        start_detection
        show_status
        ;;
    
    status)
        show_status
        ;;
    
    logs)
        show_logs "${2:-all}"
        ;;
    
    *)
        echo "用法: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "命令:"
        echo "  start   - 启动系统"
        echo "  stop    - 停止系统"
        echo "  restart - 重启系统"
        echo "  status  - 查看状态"
        echo "  logs    - 查看日志 [webui|detection|all]"
        exit 1
        ;;
esac
