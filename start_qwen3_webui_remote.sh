#!/bin/bash
# Start Qwen3 WebUI on remote OrangePi AIpro

REMOTE_HOST="HwHiAiUser@10.42.0.190"
REMOTE_DIR="/home/HwHiAiUser/ICT"
PORT=8502

echo "======================================"
echo "Starting Qwen3 WebUI on Remote Server"
echo "======================================"

# Check if already running
echo "[1/3] Checking existing processes..."
EXISTING=$(ssh $REMOTE_HOST "ps aux | grep 'streamlit run qwen3_webui_acl.py' | grep -v grep" 2>/dev/null)

if [ ! -z "$EXISTING" ]; then
    echo "WebUI is already running. Stopping..."
    ssh $REMOTE_HOST "pkill -f qwen3_webui_acl.py"
    sleep 2
fi

# Start WebUI
echo "[2/3] Starting WebUI..."
ssh $REMOTE_HOST "cd $REMOTE_DIR && nohup bash -c 'source /usr/local/Ascend/ascend-toolkit/set_env.sh && /home/HwHiAiUser/.local/bin/streamlit run qwen3_webui_acl.py --server.port $PORT --server.address 0.0.0.0' > qwen3_webui.log 2>&1 &"

sleep 5

# Check status
echo "[3/3] Checking status..."
STATUS=$(ssh $REMOTE_HOST "netstat -tuln | grep $PORT")

if [ ! -z "$STATUS" ]; then
    echo ""
    echo "======================================"
    echo "✓ WebUI started successfully!"
    echo "======================================"
    echo ""
    echo "Access URL: http://10.42.0.190:$PORT"
    echo ""
    echo "To view logs:"
    echo "  ssh $REMOTE_HOST 'tail -f $REMOTE_DIR/qwen3_webui.log'"
    echo ""
else
    echo ""
    echo "✗ Failed to start WebUI"
    echo "Check logs with:"
    echo "  ssh $REMOTE_HOST 'cat $REMOTE_DIR/qwen3_webui.log'"
    exit 1
fi
