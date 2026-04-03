#!/bin/bash
#
# MindIE 预检与启动脚本（灰度阶段）
#
# 用法:
#   bash deploy_mindie.sh /home/HwHiAiUser/ICT/config/mindie_runtime.yaml
#

set -euo pipefail

CONFIG_PATH="${1:-/home/HwHiAiUser/ICT/config/mindie_runtime.yaml}"

echo "========================================"
echo "MindIE Deploy Preflight"
echo "========================================"
echo "Config: ${CONFIG_PATH}"

if [ ! -f "${CONFIG_PATH}" ]; then
  echo "[ERROR] 配置文件不存在: ${CONFIG_PATH}"
  exit 1
fi

if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
  # shellcheck source=/dev/null
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  echo "[OK] Ascend 环境已加载"
else
  echo "[WARN] 未找到 set_env.sh，继续执行预检"
fi

if command -v npu-smi >/dev/null 2>&1; then
  echo "[OK] npu-smi 可用"
  npu-smi info | head -n 20 || true
else
  echo "[WARN] 未找到 npu-smi，请确认驱动与 CANN 安装"
fi

MINDIE_BIN=""
if command -v mindieservice_daemon >/dev/null 2>&1; then
  MINDIE_BIN="mindieservice_daemon"
elif command -v mindie-service >/dev/null 2>&1; then
  MINDIE_BIN="mindie-service"
fi

if [ -z "${MINDIE_BIN}" ]; then
  echo "[WARN] 未检测到 MindIE 可执行文件（mindieservice_daemon/mindie-service）"
  echo "[INFO] 已完成预检。请安装 MindIE 后再执行实际启动。"
  exit 0
fi

echo "[OK] 检测到 MindIE 可执行文件: ${MINDIE_BIN}"
echo "[INFO] 尝试启动（具体参数可能因版本差异需要调整）..."

set +e
"${MINDIE_BIN}" start --config "${CONFIG_PATH}"
RET=$?
set -e

if [ ${RET} -ne 0 ]; then
  echo "[WARN] 自动启动失败，建议手动查看 ${MINDIE_BIN} --help 并按版本命令启动。"
  exit ${RET}
fi

echo "[OK] MindIE 启动命令已执行。"
echo "========================================"
