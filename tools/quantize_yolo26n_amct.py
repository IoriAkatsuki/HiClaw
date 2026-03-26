#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMCT PTQ INT8 量化 YOLO26n。

执行前必须 source /usr/local/Ascend/ascend-toolkit/set_env.sh。

AMCT 正确三步流程：
  1. create_quant_config  → 生成量化配置 JSON
  2. quantize_model       → 插入量化节点，输出 modified_onnx + record_file
  3. (ORT + AMCT_SO 运行校准数据)
  4. save_model           → 将校准结果折叠到模型，输出 deploy_model.onnx
"""

from __future__ import annotations

import argparse
import glob
import shutil
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX = ROOT / "models/route_a_yolo26/yolo26n_aug_full_8419_gpu.onnx"
DEFAULT_OUTPUT = ROOT / "models/route_a_yolo26/yolo26n_aug_full_8419_gpu_int8.onnx"
DEFAULT_CALIB_DIR = ROOT / "tools/amct_calibration_data"

_AMCT_SEARCH = (
    "/usr/local/Ascend/ascend-toolkit/latest/python/site-packages",
    "/usr/local/Ascend/ascend-toolkit/*/python/site-packages",
    "/usr/local/Ascend/ascend-toolkit/latest/tools/amct/onnx",
    "/usr/local/Ascend/ascend-toolkit/*/tools/amct/onnx",
)


def _extend_sys_path() -> None:
    for pattern in _AMCT_SEARCH:
        for candidate in glob.glob(pattern):
            if Path(candidate).exists() and candidate not in sys.path:
                sys.path.insert(0, candidate)


def _import_amct():
    _extend_sys_path()
    try:
        import amct_onnx  # type: ignore
        return amct_onnx
    except ImportError as exc:
        sys.exit(
            f"[ERROR] 无法导入 amct_onnx: {exc}\n"
            "请先 source /usr/local/Ascend/ascend-toolkit/set_env.sh"
        )


def _import_ort():
    try:
        import onnxruntime as ort  # type: ignore
        return ort
    except ImportError:
        sys.exit("[ERROR] 未找到 onnxruntime，请 pip install onnxruntime")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AMCT PTQ INT8 量化 YOLO26n")
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--calib-dir", type=Path, default=DEFAULT_CALIB_DIR)
    parser.add_argument("--input-name", default="images", help="ONNX 输入节点名称")
    parser.add_argument(
        "--skip-layers",
        default="",
        help="逗号分隔的层名称模式（如 'Detect,head'），用于跳过检测头量化",
    )
    return parser.parse_args()


def _load_calibration(calib_dir: Path) -> list[Path]:
    files = sorted(calib_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"校准目录中无 .npy 文件: {calib_dir}")
    return files


def _run_calibration(modified_onnx: Path, calib_files: list[Path], input_name: str, amct) -> None:
    ort = _import_ort()
    session = ort.InferenceSession(str(modified_onnx), sess_options=amct.AMCT_SO, providers=["CPUExecutionProvider"])
    for f in calib_files:
        batch = np.load(str(f)).astype(np.float32)
        session.run(None, {input_name: batch})


def main() -> None:
    args = parse_args()
    amct = _import_amct()

    onnx_path = args.onnx.resolve()
    output_path = args.output.resolve()
    calib_dir = args.calib_dir.resolve()

    if not onnx_path.exists():
        raise FileNotFoundError(f"未找到 ONNX: {onnx_path}")
    if not calib_dir.exists():
        raise FileNotFoundError(f"校准目录不存在，先运行 prepare_amct_calibration.py: {calib_dir}")

    calib_files = _load_calibration(calib_dir)
    skip_layers = [s.strip() for s in args.skip_layers.split(",") if s.strip()]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = output_path.parent / f"{output_path.stem}_amct_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    config_path = work_dir / "quant_config.json"
    modified_onnx = work_dir / "modified_model.onnx"
    record_file = work_dir / "scale_offset_record.txt"

    print(f"[quantize] onnx={onnx_path}")
    print(f"[quantize] calib_batches={len(calib_files)}, skip_layers={skip_layers}")

    amct.create_quant_config(
        config_file=str(config_path),
        model_file=str(onnx_path),
        skip_layers=skip_layers,
        batch_num=len(calib_files),
        activation_offset=True,
    )

    amct.quantize_model(
        config_file=str(config_path),
        model_file=str(onnx_path),
        modified_onnx_file=str(modified_onnx),
        record_file=str(record_file),
    )

    print(f"[quantize] 运行校准推理 ({len(calib_files)} 批次)...")
    _run_calibration(modified_onnx, calib_files, args.input_name, amct)

    save_prefix = work_dir / output_path.stem
    amct.save_model(
        modified_onnx_file=str(modified_onnx),
        record_file=str(record_file),
        save_path=str(save_prefix),
    )

    candidates = sorted(work_dir.glob(f"{output_path.stem}*deploy*.onnx"))
    if not candidates:
        candidates = sorted(work_dir.glob(f"{output_path.stem}*.onnx"))
    if not candidates:
        raise RuntimeError(f"未找到 AMCT 输出 ONNX，请检查 {work_dir}")

    deploy_path = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copy2(str(deploy_path), str(output_path))

    orig_mb = onnx_path.stat().st_size / 1024 / 1024
    int8_mb = output_path.stat().st_size / 1024 / 1024
    print(f"[quantize] deploy_onnx={output_path}")
    print(f"[quantize] 原始={orig_mb:.1f} MB  INT8={int8_mb:.1f} MB  压缩比={int8_mb/orig_mb:.2f}")


if __name__ == "__main__":
    main()
