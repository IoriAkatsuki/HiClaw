#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_ROOT="${WORK_ROOT:-$ROOT_DIR/nightly_qwen35_0p8b}"
RUNNER="${RUNNER:-$ROOT_DIR/qwen35_nightly_atc.py}"
TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG="$WORK_ROOT/launcher_${TS}.log"
PID_FILE="$WORK_ROOT/latest.pid"

mkdir -p "$WORK_ROOT"

if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: 找不到主控脚本: $RUNNER" >&2
  exit 2
fi

nohup env PYTHONUNBUFFERED=1 python3 -u "$RUNNER" "$@" >"$LAUNCH_LOG" 2>&1 < /dev/null &
PID="$!"
printf '%s\n' "$PID" >"$PID_FILE"
ln -sfn "$LAUNCH_LOG" "$WORK_ROOT/latest_launcher.log"

sleep 2
if ! ps -p "$PID" >/dev/null 2>&1; then
  echo "ERROR: 夜间任务启动后立即退出，请检查日志: $LAUNCH_LOG" >&2
  sed -n '1,120p' "$LAUNCH_LOG" >&2 || true
  exit 3
fi

echo "已启动 Qwen3.5-0.8B 夜间 ATC 任务"
echo "PID: $PID"
echo "启动日志: $LAUNCH_LOG"
echo "主控脚本: $RUNNER"
echo "查看日志: tail -f $LAUNCH_LOG"
