#!/usr/bin/env bash
# 无人值守 VENC 探索任务：
# 1. 固化当前板端环境与系统状态
# 2. 按优先级构建/运行官方 sample 与现有 pybind 探针
# 3. 持续循环直到找到可行调用方案或达到截止时间

set -u
set -o pipefail

ICT_ROOT="${ICT_ROOT:-$HOME/ICT}"
ASCEND_ROOT_DEFAULT="$HOME/miniconda3/Ascend/cann-8.5.0"
ASCEND_ROOT="${ASCEND_ROOT:-$ASCEND_ROOT_DEFAULT}"
SET_ENV_SH="${SET_ENV_SH:-$HOME/miniconda3/Ascend/cann/set_env.sh}"
USE_SET_ENV="${USE_SET_ENV:-0}"
LOG_ROOT_BASE="${LOG_ROOT_BASE:-$ICT_ROOT/logs}"
RUN_NAME="${RUN_NAME:-venc_watchdog_$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$LOG_ROOT_BASE/$RUN_NAME}"
LATEST_LINK="$LOG_ROOT_BASE/venc_watchdog_latest"
SUMMARY_FILE="$RUN_ROOT/SUMMARY.md"
STATE_FILE="$RUN_ROOT/STATE.env"
MAIN_LOG="$RUN_ROOT/watchdog.log"
LOOP_SLEEP_SECONDS="${LOOP_SLEEP_SECONDS:-600}"
ALLOW_REBOOT="${ALLOW_REBOOT:-0}"

mkdir -p "$RUN_ROOT"
ln -sfn "$RUN_ROOT" "$LATEST_LINK"
touch "$SUMMARY_FILE" "$STATE_FILE" "$MAIN_LOG"

log() {
    local msg="[$(date '+%F %T %Z')] $*"
    printf '%s\n' "$msg" | tee -a "$MAIN_LOG"
}

append_summary() {
    printf '%s\n' "$*" >> "$SUMMARY_FILE"
}

choose_deadline() {
    if [[ -n "${DEADLINE_AT:-}" ]]; then
        printf '%s\n' "$DEADLINE_AT"
        return 0
    fi

    if [[ "$(date +%H%M%S)" -lt "070000" ]]; then
        date '+%F 07:00:00 %z'
    else
        date -d 'tomorrow 07:00:00' '+%F 07:00:00 %z'
    fi
}

DEADLINE_AT="$(choose_deadline)"
DEADLINE_EPOCH="$(date -d "$DEADLINE_AT" +%s)"

source_env() {
    if [[ "$USE_SET_ENV" = "1" && -f "$SET_ENV_SH" ]]; then
        local had_nounset=0
        if [[ "$-" == *u* ]]; then
            had_nounset=1
            set +u
        fi
        # shellcheck disable=SC1090
        source "$SET_ENV_SH"
        if [[ "$had_nounset" -eq 1 ]]; then
            set -u
        fi
    fi

    export ASCEND_ROOT
    export LD_LIBRARY_PATH="$ASCEND_ROOT/lib64:$ASCEND_ROOT/runtime/lib64:$ASCEND_ROOT/tools/aml/lib64:${LD_LIBRARY_PATH:-}"
    export LIBRARY_PATH="$ASCEND_ROOT/lib64:$ASCEND_ROOT/runtime/lib64:$ASCEND_ROOT/tools/aml/lib64:${LIBRARY_PATH:-}"
    export CPLUS_INCLUDE_PATH="$ASCEND_ROOT/runtime/include:$ASCEND_ROOT/runtime/include/acl/dvpp:${CPLUS_INCLUDE_PATH:-}"
    export PYTHONPATH="$ASCEND_ROOT/python/site-packages:$ASCEND_ROOT/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH:-}"
}

write_state() {
    {
        printf 'RUN_ROOT=%q\n' "$RUN_ROOT"
        printf 'DEADLINE_AT=%q\n' "$DEADLINE_AT"
        printf 'LAST_STEP=%q\n' "${1:-}"
        printf 'LAST_STATUS=%q\n' "${2:-}"
        printf 'LAST_UPDATE=%q\n' "$(date '+%F %T %Z')"
    } > "$STATE_FILE"
}

collect_reserved_memory() {
    python3 - <<'PY'
from pathlib import Path
import struct

base = Path('/proc/device-tree/reserved-memory')
if not base.exists():
    print('reserved-memory: not found')
    raise SystemExit(0)

for node in sorted(base.iterdir()):
    if not node.is_dir():
        continue
    name = node.name
    if (node / 'name').exists():
        try:
            name = (node / 'name').read_bytes().rstrip(b'\x00').decode('utf-8', 'ignore')
        except Exception:
            pass

    print(f'[{node.name}] name={name}')
    if (node / 'reg').exists():
        raw = (node / 'reg').read_bytes()
        vals = [struct.unpack('>Q', raw[i:i + 8])[0] for i in range(0, len(raw), 8) if len(raw[i:i + 8]) == 8]
        print(' reg=', [hex(v) for v in vals])
    if (node / 'size').exists():
        raw = (node / 'size').read_bytes()
        vals = [struct.unpack('>Q', raw[i:i + 8])[0] for i in range(0, len(raw), 8) if len(raw[i:i + 8]) == 8]
        print(' size=', [hex(v) for v in vals])
PY
}

snapshot_system() {
    local name="$1"
    local out_dir="$RUN_ROOT/$name"
    mkdir -p "$out_dir"

    {
        echo "=== date ==="
        date '+%F %T %Z'
        echo "=== uname ==="
        uname -a
        echo "=== npu-smi info ==="
        npu-smi info 2>/dev/null || true
        echo "=== npu-smi health ==="
        npu-smi info -t health -i 0 2>/dev/null || true
        echo "=== /proc/cmdline ==="
        cat /proc/cmdline
        echo "=== meminfo ==="
        grep -E 'CmaTotal|CmaFree|HugePages|AnonHugePages' /proc/meminfo || true
        echo "=== lsmod ==="
        lsmod | grep -E 'drv_venc|drv_h264e|drv_h265e|drv_vedu|drv_dvpp|drv_vpc|drv_jpege|drv_jpegd' || true
        echo "=== reserved-memory ==="
        collect_reserved_memory
        echo "=== devices ==="
        ls -l /dev/venc /dev/sys /dev/svm0 /dev/davinci0 /dev/dvpp_cmdlist 2>/dev/null || true
        echo "=== ld paths ==="
        echo "ASCEND_ROOT=$ASCEND_ROOT"
        echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    } > "$out_dir/system.txt" 2>&1

    dmesg > "$out_dir/dmesg.full.log" 2>/dev/null || true
    dmesg | tail -n 200 > "$out_dir/dmesg.tail.log" 2>/dev/null || true
}

ensure_nv12_file() {
    local path="$1"
    local width="$2"
    local height="$3"
    python3 - "$path" "$width" "$height" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
width = int(sys.argv[2])
height = int(sys.argv[3])
size = width * height * 3 // 2
path.parent.mkdir(parents=True, exist_ok=True)
if (not path.exists()) or path.stat().st_size != size:
    path.write_bytes(bytes(size))
print(path)
print(size)
PY
}

scan_outputs() {
    local base_dir="$1"
    find "$base_dir" -type f \( -name '*.264' -o -name '*.265' -o -name '*.h264' -o -name '*.h265' \) -size +0c 2>/dev/null | sort
}

run_step() {
    local step_name="$1"
    local work_dir="$2"
    local command="$3"
    local out_dir="$RUN_ROOT/$step_name"
    local before_lines=0
    local exit_code=0

    mkdir -p "$out_dir"
    before_lines="$(dmesg 2>/dev/null | wc -l || echo 0)"
    log "开始步骤: $step_name"
    write_state "$step_name" "running"

    (
        echo "=== step ==="
        echo "$step_name"
        echo "=== pwd ==="
        echo "$work_dir"
        echo "=== command ==="
        echo "$command"
        echo "=== start ==="
        date '+%F %T %Z'
        (
            cd "$work_dir"
            source_env
            timeout 1800 bash -c "$command"
        )
        exit_code=$?
        echo "=== exit_code ==="
        echo "$exit_code"
        echo "=== end ==="
        date '+%F %T %Z'
        exit "$exit_code"
    ) > "$out_dir/command.log" 2>&1 || exit_code=$?

    dmesg 2>/dev/null | tail -n +"$((before_lines + 1))" > "$out_dir/dmesg.delta.log" || true
    dmesg 2>/dev/null | tail -n 200 > "$out_dir/dmesg.tail.log" || true

    local status="soft_fail"
    local kernel_sensitive=0
    if [[ "$exit_code" -eq 0 ]]; then
        status="ok"
    fi
    case "$step_name" in
        01_*|03_*|05_*|07_*)
            kernel_sensitive=1
            ;;
    esac
    if [[ "$kernel_sensitive" -eq 1 ]] && grep -Eqs '0xa008800c|iommu_map failed -34|Create venc channel failed|error 707|507018|alloc encoder node buffer failed|Alloc encoder context failed' \
        "$out_dir/command.log" "$out_dir/dmesg.delta.log" "$out_dir/dmesg.tail.log" 2>/dev/null; then
        status="hard_fail"
    fi

    write_state "$step_name" "$status"
    log "结束步骤: $step_name status=$status exit=$exit_code"
    printf '%s\n' "$status" > "$out_dir/status.txt"
    return "$exit_code"
}

try_pybind() {
    run_step \
        "01_pybind_probe" \
        "$ICT_ROOT" \
        "export PYTHONPATH='$ICT_ROOT/pybind_venc/build':\$PYTHONPATH; python3 - <<'PY'
from venc_wrapper import VencSession
s = VencSession(640, 480, 'H264_MAIN')
print('PYBIND_VENC_OK')
s.close()
PY"
}

build_venc_image() {
    run_step \
        "02_build_venc_image" \
        "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc_image" \
        "export DDK_PATH='$ASCEND_ROOT'; \
         export NPU_HOST_LIB=\"\$DDK_PATH/lib64\"; \
         export LIBRARY_PATH=\"\$DDK_PATH/lib64:\$DDK_PATH/runtime/lib64:\$DDK_PATH/tools/aml/lib64:\${LIBRARY_PATH:-}\"; \
         export LDFLAGS='-Wl,-rpath-link,'\"\$DDK_PATH\"'/lib64 -Wl,-rpath,'\"\$DDK_PATH\"'/lib64'; \
         rm -rf build_watchdog && mkdir -p build_watchdog/intermediates/host && \
         cd build_watchdog/intermediates/host && \
         cmake ../../../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE && \
         make -j4"
}

run_venc_image() {
    ensure_nv12_file \
        "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc_image/data/dvpp_venc_128x128_nv12.yuv" \
        128 128 > "$RUN_ROOT/03_run_venc_image_input.txt"

    run_step \
        "03_run_venc_image" \
        "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc_image/out" \
        "./main 4"
}

build_venc_acl() {
    run_step \
        "04_build_venc_acl" \
        "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc" \
        "export INSTALL_DIR='$ASCEND_ROOT'; \
         export THIRDPART_PATH='$ICT_ROOT/samples_source/samples/common'; \
         export CPU_ARCH='aarch64'; \
         export LIBRARY_PATH=\"\$INSTALL_DIR/lib64:\$INSTALL_DIR/runtime/lib64:\$INSTALL_DIR/tools/aml/lib64:\${LIBRARY_PATH:-}\"; \
         export LDFLAGS='-Wl,-rpath-link,'\"\$INSTALL_DIR\"'/lib64 -Wl,-rpath,'\"\$INSTALL_DIR\"'/lib64'; \
         rm -rf build_watchdog && mkdir -p build_watchdog && \
         cd build_watchdog && \
         cmake .. && \
         make -j4"
}

run_venc_acl() {
    ensure_nv12_file \
        "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc/data/dvpp_vpc_1920x1080_nv12.yuv" \
        1920 1080 > "$RUN_ROOT/05_run_venc_acl_input.txt"

    mkdir -p "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc/scripts/output"
    run_step \
        "05_run_venc_acl" \
        "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc/scripts" \
        "../../out/main ../data/dvpp_vpc_1920x1080_nv12.yuv"
}

build_v2_himpi() {
    run_step \
        "06_build_v2_himpi" \
        "$ICT_ROOT/samples_source/samples/cplusplus/level1_single_api/7_dvpp/venc_sample" \
        "export DDK_PATH='$ASCEND_ROOT'; \
         export NPU_HOST_LIB=\"\$DDK_PATH/lib64\"; \
         export CPLUS_INCLUDE_PATH=\"\$DDK_PATH/runtime/include:\$DDK_PATH/runtime/include/acl/dvpp:\${CPLUS_INCLUDE_PATH:-}\"; \
         export LIBRARY_PATH=\"\$DDK_PATH/lib64:\$DDK_PATH/runtime/lib64:\$DDK_PATH/tools/aml/lib64:\${LIBRARY_PATH:-}\"; \
         export LDFLAGS='-Wl,-rpath-link,'\"\$DDK_PATH\"'/lib64 -Wl,-rpath,'\"\$DDK_PATH\"'/lib64'; \
         rm -rf build_watchdog && mkdir -p build_watchdog && \
         cd build_watchdog && \
         cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE && \
         make -j4"
}

run_v2_himpi() {
    ensure_nv12_file \
        "$ICT_ROOT/samples_source/samples/cplusplus/level1_single_api/7_dvpp/venc_sample/build_watchdog/dvpp_venc_128x128_nv12.yuv" \
        128 128 > "$RUN_ROOT/07_run_v2_himpi_input.txt"

    run_step \
        "07_run_v2_himpi" \
        "$ICT_ROOT/samples_source/samples/cplusplus/level1_single_api/7_dvpp/venc_sample/build_watchdog" \
        "./venc_demo --ImgWidth 128 --ImgHeight 128 --ChnNum 1 --CodecType 1 --InputFileName ./dvpp_venc_128x128_nv12.yuv --OutputFileName ./output_%d.265 --PixelFormat 1"
}

record_result_line() {
    local step_name="$1"
    local status_file="$RUN_ROOT/$step_name/status.txt"
    local result="not_run"
    if [[ -f "$status_file" ]]; then
        result="$(cat "$status_file")"
    fi
    append_summary "- $step_name: $result"
}

has_success_output() {
    {
        scan_outputs "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc_image"
        scan_outputs "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc"
        scan_outputs "$ICT_ROOT/samples_source/samples/cplusplus/level1_single_api/7_dvpp/venc_sample"
    } | grep -q .
}

mark_success_and_exit() {
    append_summary ""
    append_summary "## Final"
    append_summary "- result: found_workable_scheme"
    append_summary "- time: $(date '+%F %T %Z')"
    append_summary "- note: official sample produced encoded output."
    log "发现可行路径，结束值守。"
    exit 0
}

main_pass() {
    snapshot_system "00_baseline"
    append_summary "# VENC Watchdog"
    append_summary ""
    append_summary "- deadline: $DEADLINE_AT"
    append_summary "- ascend_root: $ASCEND_ROOT"
    append_summary "- allow_reboot: $ALLOW_REBOOT"
    append_summary ""
    append_summary "## Initial Pass"

    try_pybind || true
    record_result_line "01_pybind_probe"

    build_venc_image || true
    record_result_line "02_build_venc_image"
    if [[ -f "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc_image/out/main" ]]; then
        run_venc_image || true
        record_result_line "03_run_venc_image"
    fi

    build_venc_acl || true
    record_result_line "04_build_venc_acl"
    if [[ -f "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/out/main" ]]; then
        run_venc_acl || true
        record_result_line "05_run_venc_acl"
    fi

    build_v2_himpi || true
    record_result_line "06_build_v2_himpi"
    if [[ -f "$ICT_ROOT/samples_source/samples/cplusplus/level1_single_api/7_dvpp/venc_sample/build_watchdog/venc_demo" ]]; then
        run_v2_himpi || true
        record_result_line "07_run_v2_himpi"
    fi

    if has_success_output; then
        mark_success_and_exit
    fi
}

quick_loop() {
    local loop_id=1
    while [[ "$(date +%s)" -lt "$DEADLINE_EPOCH" ]]; do
        local round_name
        round_name="$(printf 'loop_%02d_%s' "$loop_id" "$(date +%H%M%S)")"
        snapshot_system "$round_name"

        append_summary ""
        append_summary "## Quick Loop $loop_id"
        append_summary "- time: $(date '+%F %T %Z')"

        try_pybind || true
        record_result_line "01_pybind_probe"

        if [[ -f "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc_image/out/main" ]]; then
            run_venc_image || true
            record_result_line "03_run_venc_image"
        fi

        if [[ -f "$ICT_ROOT/samples_source/samples/cplusplus/level2_simple_inference/0_data_process/out/main" ]]; then
            run_venc_acl || true
            record_result_line "05_run_venc_acl"
        fi

        if has_success_output; then
            mark_success_and_exit
        fi

        if [[ "$(date +%s)" -ge "$DEADLINE_EPOCH" ]]; then
            break
        fi

        log "未找到可行路径，休眠 ${LOOP_SLEEP_SECONDS}s 后继续。"
        sleep "$LOOP_SLEEP_SECONDS"
        loop_id=$((loop_id + 1))
    done
}

main() {
    source_env
    log "VENC Watchdog 启动，截止时间: $DEADLINE_AT"
    main_pass
    quick_loop
    append_summary ""
    append_summary "## Final"
    append_summary "- result: deadline_reached_without_success"
    append_summary "- time: $(date '+%F %T %Z')"
    log "到达截止时间，结束值守。"
}

main "$@"
