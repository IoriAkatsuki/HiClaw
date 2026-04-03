#!/usr/bin/env bash
# 启动板端无人值守 VENC 探索任务。

set -euo pipefail

BOARD_USER="${BOARD_USER:-HwHiAiUser}"
BOARD_CANDIDATES=("${BOARD_HOST:-ict.local}" "${BOARD_IP:-192.168.5.13}")
LOCAL_SCRIPT="/home/oasis/Documents/ICT/tools/board_venc_watchdog.sh"
REMOTE_SCRIPT="/home/${BOARD_USER}/ICT/tools/board_venc_watchdog.sh"
DEADLINE_AT="${DEADLINE_AT:-$(date '+%F 07:00:00 %z')}"

pick_board() {
    local candidate
    for candidate in "${BOARD_CANDIDATES[@]}"; do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "${BOARD_USER}@${candidate}" 'echo ok' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

BOARD="$(pick_board)"
SSH="ssh -o StrictHostKeyChecking=no ${BOARD_USER}@${BOARD}"

echo "使用板卡: ${BOARD_USER}@${BOARD}"
echo "截止时间: ${DEADLINE_AT}"

$SSH "mkdir -p ~/ICT/logs ~/ICT/tools"
scp -o StrictHostKeyChecking=no "$LOCAL_SCRIPT" "${BOARD_USER}@${BOARD}:${REMOTE_SCRIPT}"
$SSH "chmod +x '${REMOTE_SCRIPT}'"

PID="$($SSH "nohup env DEADLINE_AT='${DEADLINE_AT}' bash '${REMOTE_SCRIPT}' > ~/ICT/logs/venc_watchdog_launch.log 2>&1 < /dev/null & echo \$!")"

echo "已启动板端任务，PID=${PID}"
echo "板端日志: ~/ICT/logs/venc_watchdog_launch.log"
echo "板端结果目录: ~/ICT/logs/venc_watchdog_latest"
