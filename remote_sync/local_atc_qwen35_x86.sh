#!/usr/bin/env bash
set -Eeuo pipefail

# 在本机 x86_64 上执行 Qwen3.5-4B 的 ATC 编译，并把生成的 OM 回传到板卡。
# 默认假设 patched ONNX 已经同步到 local_qwen35_atc/onnx。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_WORK_DIR="${LOCAL_WORK_DIR:-$ROOT/local_qwen35_atc}"
LOCAL_ONNX_DIR="${LOCAL_ONNX_DIR:-$LOCAL_WORK_DIR/onnx}"
LOCAL_OUT_DIR="${LOCAL_OUT_DIR:-$LOCAL_WORK_DIR/out}"
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-$LOCAL_WORK_DIR/logs}"
CONDA_ENV="${CONDA_ENV:-cann85-atc}"
ENV_PREFIX="${ENV_PREFIX:-}"
MODEL_BASENAME="${MODEL_BASENAME:-qwen35_4b_seq128.atc.onnx}"
SEQ_LEN="${SEQ_LEN:-128}"
SOC_VERSION="${SOC_VERSION:-Ascend310B1}"
REMOTE_HOST="${REMOTE_HOST:-HwHiAiUser@192.168.5.13}"
REMOTE_UPLOAD_DIR="${REMOTE_UPLOAD_DIR:-/home/HwHiAiUser/ICT/x86_atc_uploads}"
REMOTE_TOKENIZER_DIR="${REMOTE_TOKENIZER_DIR:-/home/HwHiAiUser/ICT/models/qwen3.5-4b}"
RUN_SMOKE="${RUN_SMOKE:-1}"
TE_PARALLEL_COMPILER="${TE_PARALLEL_COMPILER:-1}"

mkdir -p "$LOCAL_OUT_DIR" "$LOCAL_LOG_DIR"
LOG_FILE="$LOCAL_LOG_DIR/local_atc_qwen35_x86_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  echo "[$(date '+%F %T')] $*"
}

resolve_env_prefix() {
  if [[ -n "$ENV_PREFIX" && -d "$ENV_PREFIX" ]]; then
    printf '%s\n' "$ENV_PREFIX"
    return 0
  fi

  local candidates=(
    "$HOME/.local/share/mamba/envs/$CONDA_ENV"
    "$HOME/.local/micromamba/envs/$CONDA_ENV"
    "$HOME/.conda/envs/$CONDA_ENV"
    "$HOME/miniconda3/envs/$CONDA_ENV"
  )
  local item
  for item in "${candidates[@]}"; do
    if [[ -d "$item" ]]; then
      printf '%s\n' "$item"
      return 0
    fi
  done

  if command -v mamba >/dev/null 2>&1; then
    item="$(mamba env list | awk -v env="$CONDA_ENV" '$1 == env {print $NF; exit}')"
    if [[ -n "$item" && -d "$item" ]]; then
      printf '%s\n' "$item"
      return 0
    fi
  fi

  return 1
}

resolve_set_env() {
  local prefix="$1"
  local candidates=(
    "$prefix/Ascend/ascend-toolkit/set_env.sh"
    "$prefix/Ascend/cann/set_env.sh"
  )
  local item
  for item in "${candidates[@]}"; do
    if [[ -f "$item" ]]; then
      printf '%s\n' "$item"
      return 0
    fi
  done

  item="$(find "$prefix/Ascend" -name set_env.sh -type f 2>/dev/null | head -n 1 || true)"
  if [[ -n "$item" && -f "$item" ]]; then
    printf '%s\n' "$item"
    return 0
  fi

  return 1
}

if [[ ! -f "$LOCAL_ONNX_DIR/$MODEL_BASENAME" ]]; then
  log "ERROR: 找不到 patched ONNX: $LOCAL_ONNX_DIR/$MODEL_BASENAME"
  exit 3
fi

ENV_PREFIX="$(resolve_env_prefix)"
if [[ -z "$ENV_PREFIX" || ! -d "$ENV_PREFIX" ]]; then
  log "ERROR: 无法定位本机 CANN 环境: $CONDA_ENV"
  exit 4
fi

SET_ENV_PATH="$(resolve_set_env "$ENV_PREFIX")"
if [[ -z "$SET_ENV_PATH" || ! -f "$SET_ENV_PATH" ]]; then
  log "ERROR: 在环境中找不到 set_env.sh: $ENV_PREFIX"
  exit 5
fi

log "ROOT=$ROOT"
log "LOCAL_ONNX_DIR=$LOCAL_ONNX_DIR"
log "ENV_PREFIX=$ENV_PREFIX"
log "SET_ENV_PATH=$SET_ENV_PATH"
log "TE_PARALLEL_COMPILER=$TE_PARALLEL_COMPILER"

log "验证 ATC 环境"
(
  set -Eeuo pipefail
  export PATH="$ENV_PREFIX/bin:$PATH"
  export TE_PARALLEL_COMPILER="$TE_PARALLEL_COMPILER"
  set +u
  source "$SET_ENV_PATH"
  set -u
  command -v atc
  atc --help >/dev/null
)

log "开始本机 ATC 编译"
(
  set -Eeuo pipefail
  export PATH="$ENV_PREFIX/bin:$PATH"
  export TE_PARALLEL_COMPILER="$TE_PARALLEL_COMPILER"
  set +u
  source "$SET_ENV_PATH"
  set -u
  atc \
    --model="$LOCAL_ONNX_DIR/$MODEL_BASENAME" \
    --framework=5 \
    --output="$LOCAL_OUT_DIR/qwen35_4b_seq${SEQ_LEN}_fp16" \
    --input_shape="input_ids:1,${SEQ_LEN};attention_mask:1,${SEQ_LEN};position_ids:1,${SEQ_LEN}" \
    --soc_version="$SOC_VERSION" \
    --host_env_os=linux \
    --host_env_cpu=aarch64 \
    --input_format=ND \
    --op_select_implmode=high_performance \
    --precision_mode=allow_fp32_to_fp16 \
    --log=error
)

OM_PATH="$LOCAL_OUT_DIR/qwen35_4b_seq${SEQ_LEN}_fp16.om"
if [[ ! -f "$OM_PATH" ]]; then
  log "ERROR: ATC 未生成 OM: $OM_PATH"
  exit 6
fi
log "ATC 产物: $OM_PATH"
ls -lh "$OM_PATH"

log "准备上传到板卡: $REMOTE_HOST:$REMOTE_UPLOAD_DIR"
ssh -o BatchMode=yes "$REMOTE_HOST" "mkdir -p '$REMOTE_UPLOAD_DIR'"
scp "$OM_PATH" "$REMOTE_HOST:$REMOTE_UPLOAD_DIR/"
ssh -o BatchMode=yes "$REMOTE_HOST" "ls -lh '$REMOTE_UPLOAD_DIR/$(basename "$OM_PATH")'"

if [[ "$RUN_SMOKE" == "1" ]]; then
  log "开始板卡 ACL smoke 验证"
  ssh -o BatchMode=yes "$REMOTE_HOST" \
    "python3 /home/HwHiAiUser/ICT/remote_sync/acl_smoke_qwen35.py \
      --model-path '$REMOTE_UPLOAD_DIR/$(basename "$OM_PATH")' \
      --tokenizer-path '$REMOTE_TOKENIZER_DIR' \
      --max-seq-len '$SEQ_LEN' \
      --max-new-tokens 8 \
      --prompt '你好，请用一句话介绍香橙派 AI Pro。'"
fi

log "本机 ATC + 上传流程结束，日志: $LOG_FILE"
