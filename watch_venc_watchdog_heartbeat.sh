#!/usr/bin/env bash
# 本地 heartbeat 监控：定时拉取板端 watchdog 状态，记录到本地日志。

set -euo pipefail

BOARD_USER="${BOARD_USER:-HwHiAiUser}"
BOARD_CANDIDATES=("${BOARD_HOST:-ict.local}" "${BOARD_IP:-192.168.5.13}")
POLL_SECONDS="${POLL_SECONDS:-60}"
LOCAL_ROOT="${LOCAL_ROOT:-/home/oasis/Documents/ICT/heartbeat_logs}"
RUN_NAME="${RUN_NAME:-venc_watchdog_heartbeat_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${LOCAL_ROOT}/${RUN_NAME}"
LOG_FILE="${RUN_DIR}/heartbeat.log"
STATE_SNAPSHOT="${RUN_DIR}/last_state.env"
LATEST_LINK="${LOCAL_ROOT}/venc_watchdog_heartbeat_latest"

mkdir -p "$RUN_DIR"
ln -sfn "$RUN_DIR" "$LATEST_LINK"
touch "$LOG_FILE"

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

log() {
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$LOG_FILE"
}

BOARD="$(pick_board)"
SSH="ssh -o StrictHostKeyChecking=no ${BOARD_USER}@${BOARD}"

log "heartbeat 启动，目标板卡 ${BOARD_USER}@${BOARD}，轮询 ${POLL_SECONDS}s"

last_signature=""

while true; do
    snapshot="$($SSH '
        echo "=== PROCESS ==="
        pgrep -af board_venc_watchdog.sh || true
        echo "=== STATE ==="
        cat ~/ICT/logs/venc_watchdog_latest/STATE.env 2>/dev/null || true
        echo "=== TAIL ==="
        tail -n 20 ~/ICT/logs/venc_watchdog_launch.log 2>/dev/null || true
    ' 2>&1 || true)"

    printf '%s\n' "$snapshot" > "$STATE_SNAPSHOT"

    signature="$(printf '%s\n' "$snapshot" | grep -E 'LAST_STEP=|LAST_STATUS=|LAST_UPDATE=|board_venc_watchdog.sh' | tr '\n' '|' || true)"
    if [[ "$signature" != "$last_signature" ]]; then
        log "状态变化："
        printf '%s\n' "$snapshot" >> "$LOG_FILE"
        printf '\n' >> "$LOG_FILE"
        last_signature="$signature"
    fi

    if ! printf '%s\n' "$snapshot" | grep -q 'board_venc_watchdog.sh'; then
        log "检测到板端 watchdog 已退出，heartbeat 结束。"
        break
    fi

    deadline_line="$(printf '%s\n' "$snapshot" | grep '^DEADLINE_AT=' || true)"
    if [[ -n "$deadline_line" ]]; then
        deadline_raw="${deadline_line#DEADLINE_AT=}"
        deadline_human="${deadline_raw//\\ / }"
        if [[ "$(date +%s)" -ge "$(date -d "$deadline_human" +%s 2>/dev/null || echo 0)" ]]; then
            log "达到截止时间 ${deadline_human}，heartbeat 结束。"
            break
        fi
    fi

    sleep "$POLL_SECONDS"
done
