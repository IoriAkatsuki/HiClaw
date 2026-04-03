#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/home/HwHiAiUser/ICT}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models/qwen3.5-4b}"
WORK_DIR="${WORK_DIR:-$ROOT/qwen35_autocheck}"
LOG_DIR="$WORK_DIR/logs"
PYDEPS_DIR="${PYDEPS_DIR:-$ROOT/pydeps_qwen35_clean}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$WORK_DIR/$RUN_ID"
MASTER_LOG="$LOG_DIR/${RUN_ID}.master.log"
SUMMARY_FILE="$RUN_DIR/summary.txt"
EXPORT_TIMEOUT="${EXPORT_TIMEOUT:-21600}"

mkdir -p "$LOG_DIR" "$RUN_DIR/onnx" "$RUN_DIR/om"
touch "$MASTER_LOG"
ln -sfn "$RUN_DIR" "$WORK_DIR/latest_run"
ln -sfn "$MASTER_LOG" "$WORK_DIR/latest_master.log"

exec > >(tee -a "$MASTER_LOG") 2>&1

STATUS="RUNNING"
EXPORT_STATUS="NOT_RUN"
PATCH_STATUS="NOT_RUN"
ATC_STATUS="NOT_RUN"
SMOKE_STATUS="NOT_RUN"
DEPLOY_STATUS="NOT_RUN"
SELECTED_SEQ=""
RAW_ONNX=""
PATCHED_ONNX=""
SELECTED_ONNX=""
SELECTED_OM=""
DEPLOY_OM=""

log() {
  echo "[$(date '+%F %T')] $*"
}

write_summary() {
  mkdir -p "$RUN_DIR"
  cat >"$SUMMARY_FILE" <<EOF_SUMMARY
RUN_ID=$RUN_ID
STATUS=$STATUS
MODEL_DIR=$MODEL_DIR
SELECTED_SEQ=$SELECTED_SEQ
RAW_ONNX=$RAW_ONNX
PATCHED_ONNX=$PATCHED_ONNX
SELECTED_ONNX=$SELECTED_ONNX
SELECTED_OM=$SELECTED_OM
DEPLOY_OM=$DEPLOY_OM
EXPORT_STATUS=$EXPORT_STATUS
PATCH_STATUS=$PATCH_STATUS
ATC_STATUS=$ATC_STATUS
SMOKE_STATUS=$SMOKE_STATUS
DEPLOY_STATUS=$DEPLOY_STATUS
MASTER_LOG=$MASTER_LOG
EOF_SUMMARY
}

on_exit() {
  local rc=$?
  if [[ $rc -ne 0 && "$STATUS" == "RUNNING" ]]; then
    STATUS="FAILED(rc=$rc)"
  fi
  write_summary
}

trap on_exit EXIT

log "=== Qwen3.5-4B board autocheck start ==="
log "ROOT=$ROOT"
log "RUN_ID=$RUN_ID"

if [[ ! -d "$MODEL_DIR" ]]; then
  log "ERROR: model directory not found: $MODEL_DIR"
  STATUS="FAILED"
  exit 2
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  set +u
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  set -u
else
  log "WARN: Ascend env script not found, continue with current env"
fi

log "Check NPU and ATC tool..."
npu-smi info || true
if ! command -v atc >/dev/null 2>&1; then
  log "ERROR: atc not found in PATH"
  STATUS="FAILED"
  exit 3
fi
log "atc path: $(command -v atc)"

log "Check Python export dependency set..."
set +e
PYTHONPATH="$PYDEPS_DIR" python3 - <<'PY'
from transformers import AutoConfig
import transformers
cfg = AutoConfig.from_pretrained("/home/HwHiAiUser/ICT/models/qwen3.5-4b", trust_remote_code=True)
print("transformers", transformers.__version__)
print("config", type(cfg).__name__, cfg.model_type)
PY
dep_rc=$?
set -e

if [[ $dep_rc -ne 0 ]]; then
  log "Install/update isolated deps in $PYDEPS_DIR"
  mkdir -p "$PYDEPS_DIR"
  python3 -m pip install -U --target "$PYDEPS_DIR" \
    "transformers==5.3.0" "safetensors" | tee "$LOG_DIR/${RUN_ID}.deps_install.log"
fi

log "Dependency check after install..."
PYTHONPATH="$PYDEPS_DIR" python3 - <<'PY' | tee "$LOG_DIR/${RUN_ID}.deps_check.log"
from transformers import AutoConfig
import transformers
cfg = AutoConfig.from_pretrained("/home/HwHiAiUser/ICT/models/qwen3.5-4b", trust_remote_code=True)
print("transformers", transformers.__version__)
print("config", type(cfg).__name__, cfg.model_type)
PY

log "Check ONNX environment..."
python3 - <<'PY' | tee "$LOG_DIR/${RUN_ID}.onnx_check.log"
import onnx
print("onnx", onnx.__version__)
PY

EXPORT_STATUS="RUNNING"
for seq in 128 64; do
  raw_onnx="$RUN_DIR/onnx/qwen35_4b_seq${seq}.raw.onnx"
  patched_onnx="$RUN_DIR/onnx/qwen35_4b_seq${seq}.atc.onnx"
  export_log="$LOG_DIR/${RUN_ID}.export_seq${seq}.log"
  patch_log="$LOG_DIR/${RUN_ID}.patch_seq${seq}.log"

  log "Export attempt: seq=${seq}, timeout=${EXPORT_TIMEOUT}s"
  set +e
  timeout "$EXPORT_TIMEOUT" \
    env PYTHONPATH="$PYDEPS_DIR" \
    python3 "$ROOT/remote_sync/export_qwen35_4b_torch_onnx.py" \
      --model-path "$MODEL_DIR" \
      --output "$raw_onnx" \
      --seq-len "$seq" \
      --opset 14 \
      --dtype float16 \
    2>&1 | tee "$export_log"
  export_rc=${PIPESTATUS[0]}
  set -e

  if [[ $export_rc -ne 0 || ! -s "$raw_onnx" ]]; then
    log "Export failed for seq=${seq}, rc=${export_rc}"
    continue
  fi

  EXPORT_STATUS="OK"
  PATCH_STATUS="RUNNING"
  log "Patch ONNX for ATC compatibility: seq=${seq}"
  set +e
  python3 "$ROOT/remote_sync/patch_qwen35_trilu_onnx.py" \
    --input "$raw_onnx" \
    --output "$patched_onnx" \
    2>&1 | tee "$patch_log"
  patch_rc=${PIPESTATUS[0]}
  set -e

  if [[ $patch_rc -ne 0 || ! -s "$patched_onnx" ]]; then
    PATCH_STATUS="FAILED"
    log "Patch failed for seq=${seq}, rc=${patch_rc}"
    continue
  fi

  PATCH_STATUS="OK"
  SELECTED_SEQ="$seq"
  RAW_ONNX="$raw_onnx"
  PATCHED_ONNX="$patched_onnx"
  SELECTED_ONNX="$patched_onnx"
  log "Export and patch success: $SELECTED_ONNX"
  break
done

if [[ "$EXPORT_STATUS" != "OK" ]]; then
  STATUS="FAILED"
  EXPORT_STATUS="FAILED"
  log "ERROR: all export attempts failed"
  exit 10
fi

if [[ "$PATCH_STATUS" != "OK" ]]; then
  STATUS="FAILED"
  log "ERROR: all patch attempts failed"
  exit 11
fi

ATC_STATUS="RUNNING"
SELECTED_OM="$RUN_DIR/om/qwen35_4b_seq${SELECTED_SEQ}_fp16.om"
atc_base=(
  atc
  "--model=$SELECTED_ONNX"
  "--framework=5"
  "--output=${SELECTED_OM%.om}"
  "--input_shape=input_ids:1,${SELECTED_SEQ};attention_mask:1,${SELECTED_SEQ};position_ids:1,${SELECTED_SEQ}"
  "--soc_version=Ascend310B1"
  "--input_format=ND"
  "--op_select_implmode=high_performance"
)

atc_log1="$LOG_DIR/${RUN_ID}.atc_fp16.log"
log "ATC convert attempt 1 (allow_fp32_to_fp16)"
set +e
"${atc_base[@]}" \
  --precision_mode=allow_fp32_to_fp16 \
  --log=error 2>&1 | tee "$atc_log1"
atc_rc=${PIPESTATUS[0]}
set -e

if [[ $atc_rc -ne 0 || ! -f "$SELECTED_OM" ]]; then
  atc_log2="$LOG_DIR/${RUN_ID}.atc_origin_dtype.log"
  log "ATC convert attempt 2 (must_keep_origin_dtype)"
  set +e
  "${atc_base[@]}" \
    --precision_mode=must_keep_origin_dtype \
    --log=error 2>&1 | tee "$atc_log2"
  atc_rc=${PIPESTATUS[0]}
  set -e
fi

if [[ $atc_rc -ne 0 || ! -f "$SELECTED_OM" ]]; then
  STATUS="FAILED"
  ATC_STATUS="FAILED"
  log "ERROR: ATC conversion failed"
  exit 20
fi

ATC_STATUS="OK"
log "ATC success: $SELECTED_OM"
ls -lh "$SELECTED_OM"

SMOKE_STATUS="RUNNING"
smoke_log="$LOG_DIR/${RUN_ID}.acl_smoke.log"
log "Run ACL smoke test..."
set +e
python3 "$ROOT/remote_sync/acl_smoke_qwen35.py" \
  --model-path "$SELECTED_OM" \
  --tokenizer-path "$MODEL_DIR" \
  --max-seq-len "$SELECTED_SEQ" \
  --max-new-tokens 8 \
  --prompt "你好，请用一句话介绍香橙派 AI Pro。" \
  2>&1 | tee "$smoke_log"
smoke_rc=${PIPESTATUS[0]}
set -e

if [[ $smoke_rc -ne 0 ]]; then
  STATUS="FAILED"
  SMOKE_STATUS="FAILED"
  log "ERROR: ACL smoke test failed"
  exit 30
fi

SMOKE_STATUS="OK"

DEPLOY_STATUS="RUNNING"
DEPLOY_OM="$ROOT/qwen3_5_4b_seq${SELECTED_SEQ}_fp16.om"
cp -f "$SELECTED_OM" "$DEPLOY_OM"
ln -sfn "$DEPLOY_OM" "$ROOT/qwen3_5_4b_latest.om"
DEPLOY_STATUS="OK"

STATUS="SUCCESS"
log "Deploy success: $DEPLOY_OM"
log "Latest link: $ROOT/qwen3_5_4b_latest.om"
log "=== Qwen3.5-4B board autocheck done ==="
