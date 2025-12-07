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
YOLO_MODEL="$ICT_DIR/runs/detect/train_electro61/weights/yolov8_electro61_aipp.om"
DATA_YAML="$ICT_DIR/config/electro61.yaml"
DANGER_DISTANCE=300
CONF_THRES=0.6

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
    
    # 创建WebUI服务器脚本
    cat > "$ICT_DIR/edge/unified_app/webui_server.py" << 'PYEOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WebUI HTTP服务器 - 统一检测"""
import http.server
import socketserver
import os
from pathlib import Path

PORT = 8002
WEBUI_DIR = Path.home() / 'ICT' / 'webui_http_unified'
WEBUI_DIR.mkdir(parents=True, exist_ok=True)

os.chdir(WEBUI_DIR)

# 创建默认HTML
html_file = WEBUI_DIR / 'index.html'
if not html_file.exists():
    with open(html_file, 'w') as f:
        f.write('''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能激光视觉分拣系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: "Segoe UI", Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            width: 100%;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .status-bar {
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #e9ecef;
        }
        .stat-item {
            text-align: center;
            padding: 15px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            min-width: 150px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 5px 0;
        }
        .stat-label {
            color: #6c757d;
            font-size: 0.9em;
        }
        .main-content {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 20px;
            padding: 20px;
        }
        .video-panel {
            position: relative;
        }
        .video-frame {
            width: 100%;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .info-panel {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        .info-card {
            background: white;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .info-card h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
        }
        .info-row:last-child {
            border-bottom: none;
        }
        .info-label {
            color: #6c757d;
            font-weight: 500;
        }
        .info-value {
            font-weight: bold;
            color: #495057;
        }
        .danger {
            color: #dc3545 !important;
            animation: pulse 1s infinite;
        }
        .safe {
            color: #28a745 !important;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }
        @media (max-width: 1024px) {
            .main-content {
                grid-template-columns: 1fr;
            }
            .status-bar {
                flex-wrap: wrap;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 智能激光视觉分拣系统</h1>
            <p>基于Ascend AI的实时物体检测与安全监控</p>
        </div>
        
        <div class="status-bar">
            <div class="stat-item">
                <div class="stat-label">系统状态</div>
                <div class="stat-value" id="systemStatus">--</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">帧率 (FPS)</div>
                <div class="stat-value" id="fps">--</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">检测物体</div>
                <div class="stat-value" id="objects">--</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">手部检测</div>
                <div class="stat-value" id="hands">--</div>
            </div>
        </div>
        
        <div class="main-content">
            <div class="video-panel">
                <img id="videoFrame" class="video-frame" src="frame.jpg" alt="视频流">
            </div>
            
            <div class="info-panel">
                <div class="info-card">
                    <h3>⚡ 性能指标</h3>
                    <div class="info-row">
                        <span class="info-label">YOLO推理</span>
                        <span class="info-value" id="yoloMs">-- ms</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">手部检测</span>
                        <span class="info-value" id="handMs">-- ms</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">总帧率</span>
                        <span class="info-value" id="totalFps">-- FPS</span>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>🛡️ 安全监控</h3>
                    <div class="info-row">
                        <span class="info-label">安全状态</span>
                        <span class="info-value" id="safetyStatus">--</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">最小距离</span>
                        <span class="info-value" id="minDepth">-- mm</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">危险物体</span>
                        <span class="info-value" id="dangerObj">--</span>
                    </div>
                </div>
                
                <div class="info-card">
                    <h3>ℹ️ 系统信息</h3>
                    <div class="info-row">
                        <span class="info-label">检测类别</span>
                        <span class="info-value">61类元器件</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">分辨率</span>
                        <span class="info-value">640 x 480</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">危险阈值</span>
                        <span class="info-value">300 mm</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function updateData() {
            fetch('state.json')
                .then(response => response.json())
                .then(data => {
                    // 更新状态栏
                    document.getElementById('systemStatus').textContent = data.is_danger ? '⚠️ 危险' : '✅ 安全';
                    document.getElementById('systemStatus').className = 'stat-value ' + (data.is_danger ? 'danger' : 'safe');
                    
                    document.getElementById('fps').textContent = (data.fps || 0).toFixed(1);
                    document.getElementById('objects').textContent = data.objects || 0;
                    document.getElementById('hands').textContent = data.hands || 0;
                    
                    // 更新性能指标
                    document.getElementById('yoloMs').textContent = (data.yolo_ms || 0).toFixed(1) + ' ms';
                    document.getElementById('handMs').textContent = (data.hand_ms || 0).toFixed(1) + ' ms';
                    document.getElementById('totalFps').textContent = (data.fps || 0).toFixed(1) + ' FPS';
                    
                    // 更新安全信息
                    const safetyEl = document.getElementById('safetyStatus');
                    if (data.is_danger) {
                        safetyEl.textContent = '⚠️ 危险';
                        safetyEl.className = 'info-value danger';
                    } else {
                        safetyEl.textContent = '✅ 安全';
                        safetyEl.className = 'info-value safe';
                    }
                    
                    document.getElementById('minDepth').textContent = data.min_depth_mm ? 
                        data.min_depth_mm.toFixed(0) + ' mm' : '--';
                    document.getElementById('dangerObj').textContent = data.danger_object || '无';
                    
                    // 更新视频帧
                    document.getElementById('videoFrame').src = 'frame.jpg?t=' + new Date().getTime();
                })
                .catch(err => console.error('更新失败:', err));
        }
        
        // 初始加载
        updateData();
        // 每500ms更新一次
        setInterval(updateData, 500);
    </script>
</body>
</html>''')

# 启动HTTP服务器
class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

Handler = http.server.SimpleHTTPRequestHandler
with ReusableTCPServer(("0.0.0.0", PORT), Handler) as httpd:
    print(f"WebUI服务器运行在端口 {PORT}")
    print(f"访问: http://ict.local:{PORT}")
    httpd.serve_forever()
PYEOF
    
    chmod +x "$ICT_DIR/edge/unified_app/webui_server.py"
    
    # 启动WebUI服务器
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
        echo -e "${YELLOW}  请先转换模型或指定正确的模型路径${NC}"
        return 1
    fi
    
    # 启动检测程序
    cd "$ICT_DIR/edge/unified_app"
    export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
    nohup /usr/bin/python3 unified_monitor.py \
        --yolo-model "$YOLO_MODEL" \
        --data-yaml "$DATA_YAML" \
        --danger-distance "$DANGER_DISTANCE" \
        --conf-thres "$CONF_THRES" \
        > "$LOG_DIR/detection.log" 2>&1 &
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
