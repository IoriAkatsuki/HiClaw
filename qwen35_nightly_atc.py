#!/usr/bin/env python3
"""Qwen3.5-0.8B 夜间 ATC 编译与板端验证主控脚本。"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_WORK_ROOT = ROOT / "nightly_qwen35_0p8b"
DEFAULT_REMOTE_ROOT = "/home/HwHiAiUser/ICT"
DEFAULT_CONDA_CHANNEL = "https://repo.huaweicloud.com/ascend/repos/conda"
DEFAULT_CONDA_ENV = "cann850-qwen35-nightly"
DEFAULT_PRECISION_MODES = ("allow_fp32_to_fp16", "must_keep_origin_dtype")
DEFAULT_SEQ_LENS = (64, 128)
DEFAULT_ONNX_VARIANTS = ("raw",)


@dataclass(frozen=True)
class NightlyCase:
    seq_len: int
    precision_mode: str
    onnx_variant: str

    @property
    def case_id(self) -> str:
        precision = re.sub(r"[^0-9A-Za-z_]+", "_", self.precision_mode).strip("_")
        variant = re.sub(r"[^0-9A-Za-z_]+", "_", self.onnx_variant).strip("_")
        return f"seq{self.seq_len}_{precision}_{variant}"


@dataclass
class CaseResult:
    case: NightlyCase
    status: str
    failure_kind: str | None
    om_path: str | None
    started_at: str
    finished_at: str
    log_path: str | None = None
    return_code: int | None = None


@dataclass
class NightlySummary:
    run_id: str
    deadline: dt.datetime
    finished_at: dt.datetime
    toolchain_status: str
    board_model_path: str | None
    board_onnx_path: str | None
    model_has_linear_attention: bool
    discovered_rms_norm: bool
    case_results: list[CaseResult]
    board_smoke_status: str
    successful_case_id: str | None = None
    overall_status: str = "unknown"


class MasterLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.log_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._fp.close()

    def log(self, message: str) -> None:
        line = f"[{dt.datetime.now().strftime('%F %T %Z')}] {message}"
        print(line)
        self._fp.write(line + "\n")
        self._fp.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B 夜间交叉编译与 ACL smoke 主控")
    parser.add_argument("--board-host", default=os.environ.get("BOARD_HOST", "ict.local"))
    parser.add_argument("--board-user", default=os.environ.get("BOARD_USER", "HwHiAiUser"))
    parser.add_argument("--board-root", default=os.environ.get("BOARD_ROOT", DEFAULT_REMOTE_ROOT))
    parser.add_argument("--conda-env", default=os.environ.get("CONDA_ENV", DEFAULT_CONDA_ENV))
    parser.add_argument("--conda-channel", default=os.environ.get("CONDA_CHANNEL", DEFAULT_CONDA_CHANNEL))
    parser.add_argument(
        "--toolkit-spec",
        default=os.environ.get("CANN_TOOLKIT_SPEC", "ascend::cann-toolkit==8.5.0"),
        help="conda 中 toolkit 包名",
    )
    parser.add_argument(
        "--ops-spec",
        default=os.environ.get("CANN_OPS_SPEC", ""),
        help="可选的 ops conda 包名；为空则仅安装 toolkit",
    )
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT)
    parser.add_argument("--deadline-at", default=os.environ.get("DEADLINE_AT", "09:00"))
    parser.add_argument("--seq-lens", default="64,128")
    parser.add_argument(
        "--precision-modes",
        default="allow_fp32_to_fp16,must_keep_origin_dtype",
    )
    parser.add_argument("--onnx-variants", default="raw")
    parser.add_argument("--patched-onnx", type=Path, default=None)
    parser.add_argument("--remote-model-dir", default="")
    parser.add_argument("--remote-onnx-path", default="")
    parser.add_argument("--remote-tokenizer-dir", default="")
    parser.add_argument("--timeout-minutes", type=int, default=120)
    parser.add_argument("--guard-minutes", type=int, default=20)
    parser.add_argument("--max-successful-cases", type=int, default=1)
    parser.add_argument("--soc-version", default="Ascend310B1")
    parser.add_argument("--skip-conda-create", action="store_true")
    parser.add_argument("--skip-board-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--atc-extra-args",
        default="",
        help="附加给 atc 的原始参数，空格分隔",
    )
    return parser.parse_args()


def compute_deadline(now: dt.datetime, deadline_at: str) -> dt.datetime:
    text = deadline_at.strip()
    if "T" in text or " " in text:
        try:
            return dt.datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"无法解析 deadline-at: {deadline_at}") from exc

    parts = [int(part) for part in text.split(":")]
    if len(parts) not in {2, 3}:
        raise ValueError(f"deadline-at 必须是 HH:MM 或 HH:MM:SS，当前为 {deadline_at}")

    hour, minute = parts[0], parts[1]
    second = parts[2] if len(parts) == 3 else 0
    deadline = now.replace(hour=hour, minute=minute, second=second, microsecond=0)
    if deadline <= now:
        deadline += dt.timedelta(days=1)
    return deadline


def parse_csv_list(text: str, cast=str) -> list:
    items = [item.strip() for item in text.split(",") if item.strip()]
    if cast is str:
        return items
    return [cast(item) for item in items]


def build_case_matrix(
    seq_lens: Iterable[int],
    precision_modes: Iterable[str],
    onnx_variants: Iterable[str],
) -> list[NightlyCase]:
    cases: list[NightlyCase] = []
    for seq_len in seq_lens:
        for precision_mode in precision_modes:
            for onnx_variant in onnx_variants:
                cases.append(
                    NightlyCase(
                        seq_len=int(seq_len),
                        precision_mode=str(precision_mode),
                        onnx_variant=str(onnx_variant),
                    )
                )
    return cases


def should_launch_new_case(now: dt.datetime, deadline: dt.datetime, guard_minutes: int) -> bool:
    return (deadline - now) > dt.timedelta(minutes=guard_minutes)


def classify_atc_failure(log_text: str) -> str:
    text = log_text.lower()

    if "timeout" in text or "timed out" in text:
        return "timeout"
    if any(token in text for token in ("rmsnorm", "resnorm", "addrmsnorm")):
        return "rmsnorm_unsupported"
    if any(
        token in text
        for token in (
            "linear_attn",
            "linear_attention",
            "triangular_update",
            "chunk_gated_delta_rule",
            "causal_conv1d",
            "mamba",
        )
    ):
        return "linear_attention_unsupported"
    if "trilu" in text:
        return "trilu_incompatible"
    if "not supported" in text or "unsupported" in text:
        return "generic_unsupported_op"
    if "graph engine compile failed" in text or "e69999" in text or "tbe" in text:
        return "ge_or_tbe_error"
    return "unknown"


def render_final_report(summary: NightlySummary) -> str:
    total_cases = len(summary.case_results)
    success_cases = [item for item in summary.case_results if item.status == "success"]
    failed_cases = [item for item in summary.case_results if item.status == "failed"]
    skipped_cases = [item for item in summary.case_results if item.status == "skipped"]

    lines = [
        "# Qwen3.5-0.8B 夜间 ATC 测试报告",
        "",
        "## 总览",
        f"- run_id: {summary.run_id}",
        f"- deadline: {summary.deadline.strftime('%F %T')}",
        f"- finished_at: {summary.finished_at.strftime('%F %T')}",
        f"- overall_status: {summary.overall_status}",
        f"- toolchain_status: {summary.toolchain_status}",
        f"- board_model_path: {summary.board_model_path or 'unknown'}",
        f"- board_onnx_path: {summary.board_onnx_path or 'unknown'}",
        f"- model_has_linear_attention: {'yes' if summary.model_has_linear_attention else 'no'}",
        f"- discovered_rms_norm: {'yes' if summary.discovered_rms_norm else 'no'}",
        "",
        "## 编译矩阵",
        f"- total_cases: {total_cases}",
        f"- success_cases: {len(success_cases)}",
        f"- failed_cases: {len(failed_cases)}",
        f"- skipped_cases: {len(skipped_cases)}",
        "",
        "## Case 结果",
    ]

    for item in summary.case_results:
        lines.append(
            f"- {item.case.case_id}: status={item.status}, failure_kind={item.failure_kind or 'none'}, om_path={item.om_path or 'none'}"
        )

    lines.extend(["", "## 板端验证"])
    if summary.board_smoke_status == "not_run":
        lines.append("- 未进入板端 ACL smoke")
    else:
        lines.append(f"- board_smoke_status: {summary.board_smoke_status}")

    lines.extend(["", "## 结论"])
    if success_cases:
        lines.append(
            f"- 已至少有一个 case 成功产出 OM，最佳 case: {summary.successful_case_id or success_cases[0].case.case_id}"
        )
    elif failed_cases:
        lines.append("- 所有已执行 case 均未成功产出 OM。")
        lines.append(f"- 主失败类型建议优先查看: {failed_cases[0].failure_kind or 'unknown'}")
    else:
        lines.append("- 本次未执行任何有效 case。")

    if summary.model_has_linear_attention:
        lines.append("- 模型结构仍包含 linear_attention，需持续重点关注 hybrid attention 路径。")
    if summary.discovered_rms_norm:
        lines.append("- 当前模型配置可见 RMSNorm 痕迹，但是否为主阻塞点仍需结合 ATC 日志判断。")

    return "\n".join(lines) + "\n"


def quote_remote(value: str) -> str:
    return shlex.quote(value)


def run_command(
    cmd: list[str],
    logger: MasterLogger,
    log_path: Path,
    *,
    timeout: int | None = None,
    dry_run: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    rendered = " ".join(shlex.quote(part) for part in cmd)
    logger.log(f"RUN {rendered}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        log_path.write_text(f"[dry-run] {rendered}\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=check,
        )
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + "\n" + (exc.stderr or "")
        log_path.write_text(output, encoding="utf-8")
        raise

    combined = (completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or "")
    log_path.write_text(combined, encoding="utf-8")
    return completed


def run_ssh(
    args: argparse.Namespace,
    logger: MasterLogger,
    log_path: Path,
    remote_command: str,
    *,
    timeout: int | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str]:
    target = f"{args.board_user}@{args.board_host}"
    cmd = ["ssh", "-o", "BatchMode=yes", target, remote_command]
    return run_command(cmd, logger, log_path, timeout=timeout, dry_run=dry_run)


def update_latest_link(target: Path, link_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    link_path.symlink_to(target)


def ensure_conda_toolchain(
    args: argparse.Namespace,
    logger: MasterLogger,
    run_dir: Path,
) -> tuple[str, str | None]:
    log_dir = run_dir / "logs"
    env_list_log = log_dir / "toolchain_env_list.log"
    completed = run_command(["conda", "env", "list"], logger, env_list_log, dry_run=args.dry_run)
    env_exists = args.conda_env in (completed.stdout or "")

    if not env_exists and not args.skip_conda_create:
        create_log = log_dir / "toolchain_create.log"
        run_command(
            ["conda", "create", "-n", args.conda_env, "-y", "python=3.10"],
            logger,
            create_log,
            dry_run=args.dry_run,
            check=False,
        )

    install_log = log_dir / "toolchain_install.log"
    install_cmd = [
        "conda",
        "install",
        "-n",
        args.conda_env,
        "-y",
        "-c",
        args.conda_channel,
        args.toolkit_spec,
    ]
    if args.ops_spec:
        install_cmd.append(args.ops_spec)
    install_result = run_command(install_cmd, logger, install_log, dry_run=args.dry_run)
    install_failed = install_result.returncode != 0

    if args.dry_run:
        fake_set_env = f"$CONDA_PREFIX/Ascend/ascend-toolkit/set_env.sh"
        return ("ready", fake_set_env)

    if install_failed:
        return ("failed", None)

    path_cmd = (
        "python - <<'PY'\n"
        "from pathlib import Path\n"
        "import sys\n"
        "prefix = Path(sys.prefix)\n"
        "candidates = [\n"
        "    prefix / 'Ascend' / 'ascend-toolkit' / 'set_env.sh',\n"
        "    prefix / 'Ascend' / 'cann' / 'set_env.sh',\n"
        "]\n"
        "for item in candidates:\n"
        "    if item.exists():\n"
        "        print(item)\n"
        "        raise SystemExit(0)\n"
        "print('')\n"
        "raise SystemExit(1)\n"
        "PY"
    )
    set_env_log = log_dir / "toolchain_set_env.log"
    completed = run_command(
        ["conda", "run", "-n", args.conda_env, "bash", "-lc", path_cmd],
        logger,
        set_env_log,
        dry_run=False,
    )
    set_env_path = (completed.stdout or "").strip().splitlines()
    if completed.returncode != 0 or not set_env_path or not set_env_path[-1]:
        return ("failed", None)

    verify_log = log_dir / "toolchain_verify.log"
    verify_cmd = [
        "conda",
        "run",
        "-n",
        args.conda_env,
        "bash",
        "-lc",
        f"source {shlex.quote(set_env_path[-1])} && which atc && atc --version",
    ]
    verify_result = run_command(verify_cmd, logger, verify_log, dry_run=False)
    if verify_result.returncode != 0:
        return ("failed", set_env_path[-1])

    return ("ready", set_env_path[-1])


def discover_remote_paths(
    args: argparse.Namespace,
    logger: MasterLogger,
    run_dir: Path,
) -> tuple[str | None, str | None, str | None]:
    log_dir = run_dir / "logs"
    remote_model_dir = args.remote_model_dir or ""
    if not remote_model_dir:
        remote_cmd = (
            f"find {quote_remote(args.board_root)} -type d "
            r"\( -path '*/models/qwen3.5-0.8b' -o -path '*/models/Qwen3.5-0.8B' \) "
            "| head -n 1"
        )
        completed = run_ssh(args, logger, log_dir / "discover_model_dir.log", remote_cmd, dry_run=args.dry_run)
        remote_model_dir = (completed.stdout or "").strip().splitlines()
        remote_model_dir = remote_model_dir[0] if remote_model_dir else ""

    remote_onnx_path = args.remote_onnx_path or ""
    if not remote_onnx_path:
        search_root = args.board_root
        if remote_model_dir:
            search_root = str(Path(remote_model_dir).parent)
        remote_cmd = (
            f"find {quote_remote(search_root)} -type f -iname '*.onnx' "
            r"| grep -Ei 'qwen|0\.8b|0p8b' | head -n 1"
        )
        completed = run_ssh(args, logger, log_dir / "discover_onnx.log", remote_cmd, dry_run=args.dry_run)
        remote_onnx_path = (completed.stdout or "").strip().splitlines()
        remote_onnx_path = remote_onnx_path[0] if remote_onnx_path else ""

    remote_tokenizer_dir = args.remote_tokenizer_dir or remote_model_dir or ""
    return (
        remote_model_dir or None,
        remote_onnx_path or None,
        remote_tokenizer_dir or None,
    )


def sync_remote_onnx(
    args: argparse.Namespace,
    logger: MasterLogger,
    run_dir: Path,
    remote_onnx_path: str | None,
) -> Path | None:
    if not remote_onnx_path:
        return None

    remote_dir = str(Path(remote_onnx_path).parent)
    local_dir = run_dir / "inputs" / "onnx"
    local_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "logs" / "sync_onnx.log"
    target = f"{args.board_user}@{args.board_host}:{remote_dir}/"
    run_command(
        ["rsync", "-av", target, str(local_dir) + "/"],
        logger,
        log_path,
        dry_run=args.dry_run,
    )
    return local_dir / Path(remote_onnx_path).name


def inspect_model_flags(
    args: argparse.Namespace,
    model_dir: str | None,
    logger: MasterLogger,
    run_dir: Path,
) -> tuple[bool, bool]:
    if not model_dir:
        return (False, False)

    inspect_log = run_dir / "logs" / "model_flags.log"
    remote_cmd = (
        f"python3 - {quote_remote(model_dir)} <<'PY'\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "cfg = json.loads((Path(sys.argv[1]) / 'config.json').read_text(encoding='utf-8'))\n"
        "blob = json.dumps(cfg, ensure_ascii=False).lower()\n"
        "print(json.dumps({\n"
        "  'has_linear_attention': 'linear_attention' in blob,\n"
        "  'has_rms_norm': ('rmsnorm' in blob) or ('rms_norm' in blob) or ('rms_norm_eps' in blob)\n"
        "}, ensure_ascii=False))\n"
        "PY"
    )
    completed = run_ssh(args, logger, inspect_log, remote_cmd, dry_run=args.dry_run)
    if completed.returncode != 0:
        return (False, False)
    payload = json.loads((completed.stdout or "{}").strip().splitlines()[-1])
    return (bool(payload.get("has_linear_attention")), bool(payload.get("has_rms_norm")))


def write_snapshot_files(
    args: argparse.Namespace,
    logger: MasterLogger,
    run_dir: Path,
    deadline: dt.datetime,
    remote_model_dir: str | None,
    remote_onnx_path: str | None,
) -> None:
    snapshot_dir = run_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    env_text = "\n".join(
        [
            f"host={os.uname().nodename}",
            f"cwd={ROOT}",
            f"deadline={deadline.isoformat(sep=' ')}",
            f"date={dt.datetime.now().strftime('%F %T %Z')}",
            f"conda_env={args.conda_env}",
        ]
    )
    (snapshot_dir / "env_snapshot.txt").write_text(env_text + "\n", encoding="utf-8")

    remote_cmd = (
        "echo '=== uname ==='; uname -a; "
        "echo '=== npu-smi ==='; npu-smi info 2>/dev/null || true; "
        "echo '=== system cann ==='; cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null || true"
    )
    run_ssh(args, logger, snapshot_dir / "board_snapshot.txt", remote_cmd, dry_run=args.dry_run)

    model_text = "\n".join(
        [
            f"remote_model_dir={remote_model_dir or ''}",
            f"remote_onnx_path={remote_onnx_path or ''}",
        ]
    )
    (snapshot_dir / "model_snapshot.txt").write_text(model_text + "\n", encoding="utf-8")


def build_atc_command(
    args: argparse.Namespace,
    set_env_path: str,
    onnx_path: Path,
    output_prefix: Path,
    case: NightlyCase,
) -> list[str]:
    atc_parts = [
        f"source {shlex.quote(set_env_path)}",
        "&&",
        "atc",
        f"--model={shlex.quote(str(onnx_path))}",
        "--framework=5",
        f"--output={shlex.quote(str(output_prefix))}",
        f"--input_shape=input_ids:1,{case.seq_len};attention_mask:1,{case.seq_len};position_ids:1,{case.seq_len}",
        f"--soc_version={shlex.quote(args.soc_version)}",
        "--input_format=ND",
        "--op_select_implmode=high_performance",
        f"--precision_mode={shlex.quote(case.precision_mode)}",
        "--log=error",
    ]
    extra_args = shlex.split(args.atc_extra_args)
    atc_parts.extend(extra_args)
    return ["conda", "run", "-n", args.conda_env, "bash", "-lc", " ".join(atc_parts)]


def build_matrix_summary_csv(summary_path: Path, results: list[CaseResult]) -> None:
    with summary_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(
            [
                "case_id",
                "seq_len",
                "precision_mode",
                "onnx_variant",
                "status",
                "failure_kind",
                "om_path",
                "started_at",
                "finished_at",
                "return_code",
                "log_path",
            ]
        )
        for item in results:
            writer.writerow(
                [
                    item.case.case_id,
                    item.case.seq_len,
                    item.case.precision_mode,
                    item.case.onnx_variant,
                    item.status,
                    item.failure_kind or "",
                    item.om_path or "",
                    item.started_at,
                    item.finished_at,
                    item.return_code if item.return_code is not None else "",
                    item.log_path or "",
                ]
            )


def maybe_copy_om_to_board(
    args: argparse.Namespace,
    logger: MasterLogger,
    run_dir: Path,
    case_result: CaseResult,
    tokenizer_dir: str | None,
) -> str:
    if args.skip_board_smoke or not case_result.om_path or not tokenizer_dir:
        return "not_run"

    remote_dir = f"{args.board_root}/x86_atc_uploads"
    log_dir = run_dir / "logs"
    before_log = log_dir / "board_smoke_npu_before.log"
    after_log = log_dir / "board_smoke_npu_after.log"
    upload_log = log_dir / "board_smoke_upload.log"
    smoke_log = log_dir / "board_smoke_acl.log"

    run_ssh(args, logger, before_log, "npu-smi info || true", dry_run=args.dry_run)
    run_ssh(args, logger, log_dir / "board_smoke_mkdir.log", f"mkdir -p {quote_remote(remote_dir)}", dry_run=args.dry_run)

    target = f"{args.board_user}@{args.board_host}:{remote_dir}/"
    run_command(["scp", case_result.om_path, target], logger, upload_log, dry_run=args.dry_run)

    remote_om = f"{remote_dir}/{Path(case_result.om_path).name}"
    remote_cmd = (
        f"python3 {quote_remote(args.board_root + '/remote_sync/acl_smoke_qwen35.py')} "
        f"--model-path {quote_remote(remote_om)} "
        f"--tokenizer-path {quote_remote(tokenizer_dir)} "
        f"--max-seq-len {case_result.case.seq_len} "
        f"--max-new-tokens 8 "
        f"--prompt {quote_remote('你好，请用一句话介绍香橙派 AI Pro。')}"
    )
    completed = run_ssh(args, logger, smoke_log, remote_cmd, dry_run=args.dry_run)
    run_ssh(args, logger, after_log, "npu-smi info || true", dry_run=args.dry_run)
    return "ok" if completed.returncode == 0 else "failed"


def execute_cases(
    args: argparse.Namespace,
    logger: MasterLogger,
    run_dir: Path,
    set_env_path: str,
    raw_onnx_path: Path | None,
    deadline: dt.datetime,
) -> list[CaseResult]:
    cases = build_case_matrix(
        seq_lens=parse_csv_list(args.seq_lens, int) or list(DEFAULT_SEQ_LENS),
        precision_modes=parse_csv_list(args.precision_modes, str) or list(DEFAULT_PRECISION_MODES),
        onnx_variants=parse_csv_list(args.onnx_variants, str) or list(DEFAULT_ONNX_VARIANTS),
    )
    results: list[CaseResult] = []
    success_count = 0

    variant_paths: dict[str, Path | None] = {"raw": raw_onnx_path}
    if args.patched_onnx:
        variant_paths["patched"] = args.patched_onnx

    for case in cases:
        now = dt.datetime.now()
        if not should_launch_new_case(now, deadline, args.guard_minutes):
            logger.log("已进入截止保护窗口，不再启动新的编译 case。")
            break
        if success_count >= args.max_successful_cases:
            logger.log("已达到成功 case 上限，停止继续编译。")
            break

        onnx_path = variant_paths.get(case.onnx_variant)
        started_at = dt.datetime.now()
        if onnx_path is None or not onnx_path.exists():
            results.append(
                CaseResult(
                    case=case,
                    status="skipped",
                    failure_kind="missing_onnx_variant",
                    om_path=None,
                    started_at=started_at.strftime("%F %T"),
                    finished_at=dt.datetime.now().strftime("%F %T"),
                    log_path=None,
                    return_code=None,
                )
            )
            continue

        case_dir = run_dir / "runs" / case.case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        log_path = case_dir / "atc.log"
        output_prefix = case_dir / "qwen35_0p8b"
        cmd = build_atc_command(args, set_env_path, onnx_path, output_prefix, case)

        status = "failed"
        failure_kind = "unknown"
        return_code: int | None = None
        om_path: str | None = None
        try:
            completed = run_command(
                cmd,
                logger,
                log_path,
                timeout=args.timeout_minutes * 60,
                dry_run=args.dry_run,
            )
            return_code = completed.returncode
            om_candidate = output_prefix.with_suffix(".om")
            log_text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path.exists() else ""
            if completed.returncode == 0 and (args.dry_run or om_candidate.exists()):
                status = "success"
                failure_kind = None
                om_path = str(om_candidate)
                success_count += 1
            else:
                failure_kind = classify_atc_failure(log_text)
        except subprocess.TimeoutExpired:
            timeout_text = "[runner] timeout\n"
            log_path.write_text(timeout_text, encoding="utf-8")
            status = "failed"
            failure_kind = "timeout"

        finished_at = dt.datetime.now()
        meta = {
            "case": asdict(case),
            "status": status,
            "failure_kind": failure_kind,
            "om_path": om_path,
            "started_at": started_at.strftime("%F %T"),
            "finished_at": finished_at.strftime("%F %T"),
            "return_code": return_code,
        }
        (case_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append(
            CaseResult(
                case=case,
                status=status,
                failure_kind=failure_kind,
                om_path=om_path,
                started_at=started_at.strftime("%F %T"),
                finished_at=finished_at.strftime("%F %T"),
                log_path=str(log_path),
                return_code=return_code,
            )
        )

    build_matrix_summary_csv(run_dir / "matrix_summary.csv", results)
    return results


def select_successful_case(results: list[CaseResult]) -> CaseResult | None:
    for item in results:
        if item.status == "success":
            return item
    return None


def main() -> int:
    args = parse_args()
    now = dt.datetime.now()
    deadline = compute_deadline(now, args.deadline_at)
    run_id = now.strftime("%Y%m%d_%H%M%S")
    run_dir = args.work_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "master.log"
    logger = MasterLogger(log_path)
    update_latest_link(run_dir, args.work_root / "latest_run")
    update_latest_link(log_path, args.work_root / "latest_master.log")

    try:
        logger.log("Qwen3.5-0.8B 夜间任务启动")
        logger.log(f"run_id={run_id}")
        logger.log(f"deadline={deadline.strftime('%F %T')}")

        remote_model_dir, remote_onnx_path, remote_tokenizer_dir = discover_remote_paths(args, logger, run_dir)
        write_snapshot_files(args, logger, run_dir, deadline, remote_model_dir, remote_onnx_path)

        toolchain_status, set_env_path = ensure_conda_toolchain(args, logger, run_dir)
        if toolchain_status != "ready" or not set_env_path:
            summary = NightlySummary(
                run_id=run_id,
                deadline=deadline,
                finished_at=dt.datetime.now(),
                toolchain_status=toolchain_status,
                board_model_path=remote_model_dir,
                board_onnx_path=remote_onnx_path,
                model_has_linear_attention=False,
                discovered_rms_norm=False,
                case_results=[],
                board_smoke_status="not_run",
                overall_status="toolchain_failed",
            )
            (run_dir / "nightly_final_report.md").write_text(render_final_report(summary), encoding="utf-8")
            logger.log("工具链未就绪，任务结束。")
            return 3

        has_linear_attention, has_rms_norm = inspect_model_flags(args, remote_model_dir, logger, run_dir)
        local_raw_onnx = sync_remote_onnx(args, logger, run_dir, remote_onnx_path)
        case_results = execute_cases(args, logger, run_dir, set_env_path, local_raw_onnx, deadline)
        successful_case = select_successful_case(case_results)
        board_smoke_status = maybe_copy_om_to_board(args, logger, run_dir, successful_case, remote_tokenizer_dir) if successful_case else "not_run"

        overall_status = "success" if successful_case and board_smoke_status in {"ok", "not_run"} else "compiled_failed"
        summary = NightlySummary(
            run_id=run_id,
            deadline=deadline,
            finished_at=dt.datetime.now(),
            toolchain_status=toolchain_status,
            board_model_path=remote_model_dir,
            board_onnx_path=remote_onnx_path,
            model_has_linear_attention=has_linear_attention,
            discovered_rms_norm=has_rms_norm,
            case_results=case_results,
            board_smoke_status=board_smoke_status,
            successful_case_id=successful_case.case.case_id if successful_case else None,
            overall_status=overall_status,
        )
        report_path = run_dir / "nightly_final_report.md"
        report_path.write_text(render_final_report(summary), encoding="utf-8")
        logger.log(f"最终报告已生成: {report_path}")
        return 0 if successful_case else 2
    finally:
        logger.close()


if __name__ == "__main__":
    raise SystemExit(main())
