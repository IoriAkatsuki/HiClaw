#!/usr/bin/env bash
# 拉取板端最新的 VENC Watchdog 日志。

set -euo pipefail

BOARD_USER="${BOARD_USER:-HwHiAiUser}"
BOARD_CANDIDATES=("${BOARD_HOST:-ict.local}" "${BOARD_IP:-192.168.5.13}")
LOCAL_ROOT="/home/oasis/Documents/ICT/board_logs/venc_watchdog_$(date +%Y%m%d_%H%M%S)"

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

mkdir -p "$LOCAL_ROOT"

REMOTE_LATEST="$($SSH 'readlink -f ~/ICT/logs/venc_watchdog_latest 2>/dev/null || true')"
if [[ -z "$REMOTE_LATEST" ]]; then
    echo "未找到板端 venc_watchdog_latest"
    exit 1
fi

echo "使用板卡: ${BOARD_USER}@${BOARD}"
echo "拉取目录: ${REMOTE_LATEST}"
echo "本地目录: ${LOCAL_ROOT}"

rsync -avz -e "ssh -o StrictHostKeyChecking=no" \
  "${BOARD_USER}@${BOARD}:${REMOTE_LATEST}/" \
  "${LOCAL_ROOT}/"

$SSH 'tail -n 80 ~/ICT/logs/venc_watchdog_launch.log 2>/dev/null || true' > "${LOCAL_ROOT}/launch.log"

echo "完成。可查看:"
echo "  ${LOCAL_ROOT}/SUMMARY.md"
echo "  ${LOCAL_ROOT}/watchdog.log"
