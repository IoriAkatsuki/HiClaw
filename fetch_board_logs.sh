#!/bin/bash
# 从板卡拉取日志和运行状态
BOARD="${BOARD_IP:-192.168.5.13}"
BOARD_USER="${BOARD_USER:-HwHiAiUser}"
LOCAL_LOG_DIR="./board_logs/$(date +%Y%m%d_%H%M%S)"
SSH="ssh -o StrictHostKeyChecking=no $BOARD_USER@$BOARD"

mkdir -p "$LOCAL_LOG_DIR"

echo "=== 板卡日志回传 ($BOARD) ==="
echo "本地存储: $LOCAL_LOG_DIR"
echo ""

# 1. 系统信息
echo "[1/5] 系统信息..."
$SSH 'echo "=== uname ==="; uname -a; echo "=== npu-smi ==="; npu-smi info 2>/dev/null || echo "npu-smi not found"; echo "=== CANN version ==="; cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || echo "no system cann"; ls ~/miniconda3/Ascend/cann/set_env.sh 2>/dev/null && echo "conda cann found" || echo "no conda cann"' \
  > "$LOCAL_LOG_DIR/system_info.txt" 2>&1

# 2. 统一监控日志
echo "[2/5] 统一监控日志..."
$SSH 'cat /tmp/unified_start.log 2>/dev/null' \
  > "$LOCAL_LOG_DIR/unified_start.log" 2>&1

# 3. WebUI 日志
echo "[3/5] WebUI 日志..."
$SSH 'cat ~/ICT/logs/webui_unified.log 2>/dev/null' \
  > "$LOCAL_LOG_DIR/webui_unified.log" 2>&1

# 4. 运行状态快照
echo "[4/5] 运行状态..."
$SSH 'curl -s http://localhost:8002/state.json 2>/dev/null' \
  > "$LOCAL_LOG_DIR/state.json" 2>&1

# 5. conda / CANN 安装日志
echo "[5/5] CANN 安装状态..."
$SSH 'export PATH=$HOME/miniconda3/bin:$PATH 2>/dev/null; conda list 2>/dev/null | grep -E "(cann|ascend)"; echo "=== venc test ==="; source $HOME/miniconda3/Ascend/cann/set_env.sh 2>/dev/null; export PYTHONPATH=$HOME/ICT/pybind_venc/build:$PYTHONPATH; python3 -c "from venc_wrapper import VencSession; s=VencSession(640,480); print(\"VENC OK\"); s.close()" 2>&1 || echo "VENC test skipped or failed"' \
  > "$LOCAL_LOG_DIR/cann_status.txt" 2>&1

echo ""
echo "=== 完成 ==="
echo "日志已保存到: $LOCAL_LOG_DIR"
ls -la "$LOCAL_LOG_DIR/"
