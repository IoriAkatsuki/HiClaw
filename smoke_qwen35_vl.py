#!/usr/bin/env python3
"""Ascend 310B1 上的 Qwen3.5-0.8B 最小多模态 PoC。"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.utils.dummy_torchvision_objects import BaseVideoProcessor as DummyTorchvisionBaseVideoProcessor

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None


PROBE_CHOICES = ("full_generate", "text_prefill", "mm_prefill", "mm_decode1")


def build_messages(
    question: str,
    image: Image.Image | None = None,
    include_image: bool = True,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if include_image:
        image_part: dict[str, Any] = {"type": "image"}
        if image is not None:
            image_part["image"] = image
        content.append(image_part)
    content.append({"type": "text", "text": question})
    return [{"role": "user", "content": content}]


def resize_image_if_needed(image: Image.Image, max_edge: int) -> Image.Image:
    image = image.convert("RGB")
    longest = max(image.size)
    if longest <= max_edge:
        return image
    scale = max_edge / float(longest)
    new_size = (
        max(1, int(round(image.width * scale))),
        max(1, int(round(image.height * scale))),
    )
    return image.resize(new_size, Image.Resampling.LANCZOS)


def classify_runtime_issue(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "unsupported op" in text or "unsupported operator" in text:
        return "unsupported_op"
    if (
        "dtype" in text
        or "bfloat16" in text
        or "float32" in text
        or "float16" in text
        or "c10::half" in text
        or ("input type" in text and "bias type" in text and "should be the same" in text)
    ):
        return "dtype"
    if "processor mismatch" in text or ("processor" in text and "mismatch" in text):
        return "processor_mismatch"
    if "out of memory" in text or "oom" in text or "memory alloc" in text:
        return "oom"
    if "timeout" in text or "timed out" in text or "hang" in text:
        return "timeout"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5-0.8B 最小多模态单图 PoC")
    parser.add_argument("--model-path", required=True, help="本地 Hugging Face 模型目录")
    parser.add_argument("--image-path", required=True, help="输入图片路径（单张 JPG/PNG）")
    parser.add_argument("--question", required=True, help="中文问题")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-image-edge", type=int, default=384)
    parser.add_argument("--failure-report", default="failure_report.md")
    parser.add_argument("--metrics-json", default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--probe", choices=PROBE_CHOICES, default="full_generate")
    parser.add_argument("--probe-timeout", type=int, default=180)
    parser.add_argument("--disable-cache", action="store_true", help="诊断用：禁用 use_cache")
    parser.add_argument("--child-probe", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_numpy_compatible() -> None:
    major = int(str(np.__version__).split('.', 1)[0])
    if major >= 2:
        raise RuntimeError(f"numpy 版本不兼容: {np.__version__}，当前 CANN/torch_npu 需要 numpy<2")


def get_device() -> torch.device:
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu 不可用，说明 torch_npu 未正确加载")
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() 为 False")
    torch.npu.set_device(0)
    return torch.device("npu:0")


def select_model_loader(model_path: str):
    config_path = Path(model_path) / "config.json"
    architecture = ""
    if config_path.exists():
        config = json.loads(config_path.read_text())
        architectures = config.get("architectures") or []
        if architectures:
            architecture = architectures[0]
    if "ConditionalGeneration" in architecture:
        return AutoModelForImageTextToText
    if AutoModelForVision2Seq is not None:
        return AutoModelForVision2Seq
    return AutoModelForImageTextToText


class PlaceholderVideoProcessor(DummyTorchvisionBaseVideoProcessor):
    """无 torchvision 时给 Qwen3VLProcessor 提供的占位 video processor。"""

    model_input_names: list[str] = []

    def __init__(self) -> None:
        self._processor_class = "Qwen3VLProcessor"
        self.model_valid_processing_keys = []
        self.merge_size = 2

    def __call__(self, videos, **kwargs):
        raise RuntimeError("image-only PoC 不应调用 video_processor")


def should_use_image_only_processor_fallback(model_path: str, exc: BaseException) -> bool:
    text = str(exc)
    if isinstance(exc, ImportError):
        return "AutoVideoProcessor requires the Torchvision library" in text
    if not isinstance(exc, TypeError):
        return False
    if "argument of type 'NoneType' is not iterable" not in text:
        return False
    preprocessor_path = Path(model_path) / "preprocessor_config.json"
    if not preprocessor_path.exists():
        return False
    try:
        config = json.loads(preprocessor_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return config.get("processor_class") == "Qwen3VLProcessor"


def build_image_only_processor_fallback(model_path: str):
    image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    chat_template_path = Path(model_path) / "chat_template.jinja"
    chat_template = None
    if chat_template_path.exists():
        chat_template = chat_template_path.read_text(encoding="utf-8")
    return Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=PlaceholderVideoProcessor(),
        chat_template=chat_template,
    )


def load_processor(model_path: str):
    try:
        return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    except (ImportError, TypeError) as exc:
        if should_use_image_only_processor_fallback(model_path, exc):
            return build_image_only_processor_fallback(model_path)
        raise


def load_processor_and_model(model_path: str, device: torch.device, move_to_device: bool = True):
    processor = load_processor(model_path)
    loader = select_model_loader(model_path)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": torch.float16,
    }
    try:
        model = loader.from_pretrained(model_path, attn_implementation="eager", **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        try:
            model = loader.from_pretrained(model_path, attn_implementation="eager", **kwargs)
        except TypeError:
            model = loader.from_pretrained(model_path, **kwargs)
    model.eval()
    if move_to_device:
        model.to(device)
    return processor, model


def apply_chat_template(
    processor,
    image: Image.Image | None,
    question: str,
    include_image: bool = True,
) -> tuple[str, dict[str, Any]]:
    messages = build_messages(question, image=image, include_image=include_image)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    processor_kwargs: dict[str, Any] = {
        "text": [prompt],
        "return_tensors": "pt",
        "padding": True,
    }
    if include_image:
        processor_kwargs["images"] = [image]
    inputs = processor(**processor_kwargs)
    seq_len = int(inputs["input_ids"].shape[1])
    return prompt, inputs | {"_prompt_seq_len": seq_len}


def get_model_dtype(model) -> torch.dtype:
    return next(model.parameters()).dtype


def get_patch_embed_proj(model):
    if hasattr(model, "visual") and hasattr(model.visual, "patch_embed"):
        return model.visual.patch_embed.proj
    if hasattr(model, "model") and hasattr(model.model, "visual") and hasattr(model.model.visual, "patch_embed"):
        return model.model.visual.patch_embed.proj
    raise RuntimeError("processor mismatch: 无法定位 patch_embed.proj")


def probe_patch_embed_proj_dtype(model) -> dict[str, Any]:
    proj = get_patch_embed_proj(model)
    return {
        "weight_dtype": str(proj.weight.dtype),
        "bias_dtype": None if proj.bias is None else str(proj.bias.dtype),
        "weight_device": str(proj.weight.device),
        "bias_device": None if proj.bias is None else str(proj.bias.device),
    }


def print_patch_embed_proj_probe(label: str, probe: dict[str, Any]) -> None:
    print(f"[patch-embed-probe] {label}", file=sys.stderr)
    print(
        "[patch-embed-probe] "
        f"weight dtype={probe['weight_dtype']}, device={probe['weight_device']}; "
        f"bias dtype={probe['bias_dtype']}, device={probe['bias_device']}",
        file=sys.stderr,
    )


def normalize_patch_embed_proj_dtype(model) -> dict[str, dict[str, Any]]:
    proj = get_patch_embed_proj(model)
    before = probe_patch_embed_proj_dtype(model)
    with torch.no_grad():
        if proj.bias is not None and proj.bias.dtype != proj.weight.dtype:
            proj.bias.data = proj.bias.data.to(device=proj.weight.device, dtype=proj.weight.dtype)
    after = probe_patch_embed_proj_dtype(model)
    return {"before": before, "after": after}


def force_patch_embed_proj_fp16_after_to_device(model) -> dict[str, dict[str, Any]]:
    proj = get_patch_embed_proj(model)
    before = probe_patch_embed_proj_dtype(model)
    with torch.no_grad():
        proj.weight.data = proj.weight.data.to(device=proj.weight.device, dtype=torch.float16)
        if proj.bias is not None:
            proj.bias.data = proj.bias.data.to(device=proj.weight.device, dtype=torch.float16)
    after = probe_patch_embed_proj_dtype(model)
    return {"before": before, "after": after}


def find_mixed_param_modules(model, limit: int = 20) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        dtypes = sorted({str(param.dtype) for _, param in module.named_parameters(recurse=False)})
        if len(dtypes) > 1:
            found.append({"name": name, "dtypes": dtypes})
            if len(found) >= limit:
                break
    return found


def move_inputs_to_device_and_dtype_by_module(inputs: dict[str, Any], module, device: torch.device) -> dict[str, Any]:
    target_dtype = module.weight.dtype
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if key.startswith("_"):
            moved[key] = value
        elif hasattr(value, "to"):
            if getattr(value, "dtype", None) is not None and value.dtype.is_floating_point:
                moved[key] = value.to(device=device, dtype=target_dtype, non_blocking=True)
            else:
                moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def collect_tensor_debug_info(inputs: dict[str, Any]) -> dict[str, dict[str, Any]]:
    info: dict[str, dict[str, Any]] = {}
    for key, value in inputs.items():
        if key.startswith("_") or not hasattr(value, "dtype"):
            continue
        shape = list(value.shape) if hasattr(value, "shape") else []
        device = str(value.device) if hasattr(value, "device") else "unknown"
        stride = list(value.stride()) if hasattr(value, "stride") else []
        is_contiguous = bool(value.is_contiguous()) if hasattr(value, "is_contiguous") else None
        storage_offset = int(value.storage_offset()) if hasattr(value, "storage_offset") else None
        info[key] = {
            "dtype": str(value.dtype),
            "shape": shape,
            "device": device,
            "stride": stride,
            "is_contiguous": is_contiguous,
            "storage_offset": storage_offset,
        }
    return info


def print_tensor_debug_info(label: str, info: dict[str, dict[str, Any]]) -> None:
    print(f"[dtype-debug] {label}", file=sys.stderr)
    for key, value in info.items():
        print(
            "[dtype-debug] "
            f"{key}: dtype={value['dtype']}, shape={tuple(value['shape'])}, device={value['device']}, "
            f"stride={tuple(value['stride'])}, is_contiguous={value['is_contiguous']}, "
            f"storage_offset={value['storage_offset']}",
            file=sys.stderr,
        )


def build_decode_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    append_mask = torch.ones(
        (attention_mask.shape[0], 1),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat([attention_mask, append_mask], dim=1)


def extract_first_kv_debug_info(past_key_values) -> dict[str, Any]:
    if not past_key_values:
        return {"available": False}
    first_layer = past_key_values[0]
    if not isinstance(first_layer, (tuple, list)) or not first_layer:
        return {"available": False}
    first_kv = first_layer[0]
    if not hasattr(first_kv, "dtype"):
        return {"available": False}
    return {
        "available": True,
        "dtype": str(first_kv.dtype),
        "shape": list(first_kv.shape),
        "device": str(first_kv.device),
    }


def format_probe_stage_marker(stage: str, payload: dict[str, Any] | None = None) -> str:
    if payload:
        return f"[probe-stage] {stage} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}"
    return f"[probe-stage] {stage}"


def get_npu_memory_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    if not hasattr(torch, "npu"):
        return snapshot
    for name in [
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ]:
        fn = getattr(torch.npu, name, None)
        if callable(fn):
            try:
                snapshot[name] = int(fn())
            except Exception:
                snapshot[name] = None
    return snapshot


def bytes_to_mib(value: int | None) -> float | None:
    if value is None:
        return None
    return round(value / 1024 / 1024, 2)


def run_npu_smi() -> dict[str, Any]:
    result: dict[str, Any] = {"raw": "", "parsed": {}}
    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        result["raw"] = "npu-smi not found"
        return result
    result["raw"] = completed.stdout + completed.stderr
    pattern = re.compile(r"(?i)(memory|hbm)[^\n]*?(\d+)\s*/\s*(\d+)\s*(mib|mb|gib|gb)")
    match = pattern.search(result["raw"])
    if match:
        result["parsed"] = {
            "label": match.group(1),
            "used": int(match.group(2)),
            "total": int(match.group(3)),
            "unit": match.group(4),
        }
    return result


def strip_thinking_markers(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S).strip()


def write_failure_report(path: str, args: argparse.Namespace, exc: BaseException) -> None:
    trace = traceback.format_exc()
    issue = classify_runtime_issue(exc)
    content = f"""# failure_report

## 失败分类
- `{issue}`

## 运行参数
- `model_path`: `{args.model_path}`
- `image_path`: `{args.image_path}`
- `question`: `{args.question}`
- `probe`: `{args.probe}`
- `max_new_tokens`: `{args.max_new_tokens}`
- `max_context_len`: `{args.max_context_len}`
- `max_image_edge`: `{args.max_image_edge}`
- `disable_cache`: `{args.disable_cache}`
- `use_cache`: `{not args.disable_cache}`

## 最小复现命令
```bash
python3 smoke_qwen35_vl.py \\
  --model-path {args.model_path} \\
  --image-path {args.image_path} \\
  --question {json.dumps(args.question, ensure_ascii=False)} \\
  --probe {args.probe} \\
  --max-new-tokens {args.max_new_tokens} \\
  --max-context-len {args.max_context_len} \\
  --max-image-edge {args.max_image_edge}
```

## 完整错误栈
```text
{trace}
```
"""
    Path(path).write_text(content)


def build_base_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_path": args.model_path,
        "image_path": args.image_path,
        "question": args.question,
        "probe": args.probe,
        "max_new_tokens": args.max_new_tokens,
        "max_context_len": args.max_context_len,
        "max_image_edge": args.max_image_edge,
        "disable_cache": bool(args.disable_cache),
        "use_cache": not bool(args.disable_cache),
    }


def tail_text(text: str | None, limit: int = 4000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[-limit:]


def write_payload_failure_report(path: str, args: argparse.Namespace, payload: dict[str, Any]) -> None:
    content = f"""# failure_report

## 失败分类
- `{payload.get('failure_type', 'unknown')}`

## 运行参数
- `model_path`: `{args.model_path}`
- `image_path`: `{args.image_path}`
- `question`: `{args.question}`
- `probe`: `{args.probe}`
- `max_new_tokens`: `{args.max_new_tokens}`
- `max_context_len`: `{args.max_context_len}`
- `max_image_edge`: `{args.max_image_edge}`
- `disable_cache`: `{args.disable_cache}`
- `use_cache`: `{not args.disable_cache}`

## 失败说明
- `error_type`: `{payload.get('error_type', 'UnknownError')}`
- `error`: `{payload.get('error', '')}`

## stdout tail
```text
{payload.get('stdout_tail', '')}
```

## stderr tail
```text
{payload.get('stderr_tail', '')}
```
"""
    Path(path).write_text(content)


def build_child_probe_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-path",
        args.model_path,
        "--image-path",
        args.image_path,
        "--question",
        args.question,
        "--probe",
        args.probe,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--max-context-len",
        str(args.max_context_len),
        "--max-image-edge",
        str(args.max_image_edge),
        "--failure-report",
        args.failure_report,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--child-probe",
    ]
    if args.disable_cache:
        command.append("--disable-cache")
    return command


def synchronize_npu() -> None:
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()


def reset_npu_peak_stats() -> None:
    if hasattr(torch, "npu") and hasattr(torch.npu, "reset_peak_memory_stats"):
        torch.npu.reset_peak_memory_stats()
    synchronize_npu()


def collect_runtime_metrics() -> dict[str, Any]:
    npu_mem = get_npu_memory_snapshot()
    return {
        "host_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
        "npu_memory_bytes": npu_mem,
        "npu_memory_mib": {key: bytes_to_mib(value) for key, value in npu_mem.items()},
        "npu_smi": run_npu_smi(),
    }


def prepare_runtime_context(args: argparse.Namespace, include_image: bool) -> dict[str, Any]:
    ensure_numpy_compatible()
    device = get_device()
    processor, model = load_processor_and_model(args.model_path, device, move_to_device=False)
    first_param_dtype = str(get_model_dtype(model))
    patch_embed_probe_before_to_device = probe_patch_embed_proj_dtype(model)
    print_patch_embed_proj_probe("after_from_pretrained", patch_embed_probe_before_to_device)

    model.to(device)
    patch_embed_probe_after_to_device = probe_patch_embed_proj_dtype(model)
    print_patch_embed_proj_probe("after_to_device", patch_embed_probe_after_to_device)
    mixed_param_modules_after_to_device = find_mixed_param_modules(model)
    patch_embed_probe_after_force_fp16 = force_patch_embed_proj_fp16_after_to_device(model)
    print_patch_embed_proj_probe(
        "after_force_patch_embed_proj_fp16_before",
        patch_embed_probe_after_force_fp16["before"],
    )
    print_patch_embed_proj_probe(
        "after_force_patch_embed_proj_fp16_after",
        patch_embed_probe_after_force_fp16["after"],
    )
    mixed_param_modules_after_force_fp16 = find_mixed_param_modules(model)

    image = None
    if include_image:
        with Image.open(args.image_path) as source_image:
            image = resize_image_if_needed(source_image, args.max_image_edge)

    prompt, inputs = apply_chat_template(processor, image, args.question, include_image=include_image)
    prompt_seq_len = inputs.pop("_prompt_seq_len")
    if prompt_seq_len > args.max_context_len:
        raise RuntimeError(f"processor mismatch: 输入长度 {prompt_seq_len} 超过限制 {args.max_context_len}")

    input_tensor_info_before_move = collect_tensor_debug_info(inputs)
    print(f"[dtype-debug] model first param dtype: {first_param_dtype}", file=sys.stderr)
    print_tensor_debug_info("before_move", input_tensor_info_before_move)
    inputs = move_inputs_to_device_and_dtype_by_module(inputs, get_patch_embed_proj(model), device)
    input_tensor_info_after_move = collect_tensor_debug_info(inputs)
    print_tensor_debug_info("after_move", input_tensor_info_after_move)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("processor mismatch: processor.tokenizer 不存在")

    return {
        "device": device,
        "processor": processor,
        "tokenizer": tokenizer,
        "model": model,
        "image": image,
        "prompt": prompt,
        "inputs": inputs,
        "prompt_seq_len": prompt_seq_len,
        "model_first_param_dtype": first_param_dtype,
        "patch_embed_probe_before_to_device": patch_embed_probe_before_to_device,
        "patch_embed_probe_after_to_device": patch_embed_probe_after_to_device,
        "patch_embed_probe_after_force_fp16": patch_embed_probe_after_force_fp16,
        "mixed_param_modules_after_to_device": mixed_param_modules_after_to_device,
        "mixed_param_modules_after_force_fp16": mixed_param_modules_after_force_fp16,
        "input_tensor_info_before_move": input_tensor_info_before_move,
        "input_tensor_info_after_move": input_tensor_info_after_move,
    }


def add_runtime_payload(payload: dict[str, Any], runtime: dict[str, Any]) -> None:
    payload.update(
        {
            "model_first_param_dtype": runtime["model_first_param_dtype"],
            "patch_embed_probe_before_to_device": runtime["patch_embed_probe_before_to_device"],
            "patch_embed_probe_after_to_device": runtime["patch_embed_probe_after_to_device"],
            "patch_embed_probe_after_force_fp16": runtime["patch_embed_probe_after_force_fp16"],
            "mixed_param_modules_after_to_device": runtime["mixed_param_modules_after_to_device"],
            "mixed_param_modules_after_force_fp16": runtime["mixed_param_modules_after_force_fp16"],
            "input_tensor_info_before_move": runtime["input_tensor_info_before_move"],
            "input_tensor_info_after_move": runtime["input_tensor_info_after_move"],
            "prompt_length": runtime["prompt_seq_len"],
            "prompt_preview": runtime["prompt"][:200],
            "device": str(runtime["device"]),
        }
    )
    if runtime["image"] is not None:
        payload["resized_image_size"] = list(runtime["image"].size)


def execute_text_prefill(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    use_cache = not args.disable_cache
    stage_start = time.perf_counter()
    print(
        format_probe_stage_marker("text_prefill_enter_model_call", {"use_cache": use_cache}),
        file=sys.stderr,
    )
    with torch.inference_mode():
        prefill = runtime["model"](
            input_ids=runtime["inputs"]["input_ids"],
            attention_mask=runtime["inputs"]["attention_mask"],
            use_cache=use_cache,
            return_dict=True,
        )
    synchronize_npu()
    print(
        format_probe_stage_marker(
            "text_prefill_done",
            {"has_past_key_values": prefill.past_key_values is not None},
        ),
        file=sys.stderr,
    )
    return {
        "status": "ok",
        "probe_stage": "text_prefill",
        "prefill_seconds": round(time.perf_counter() - stage_start, 4),
        "prefill_logits_shape": list(prefill.logits.shape),
        "has_past_key_values": prefill.past_key_values is not None,
        "first_kv_debug": extract_first_kv_debug_info(prefill.past_key_values),
    }


def execute_mm_prefill(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    use_cache = not args.disable_cache
    stage_start = time.perf_counter()
    print(
        format_probe_stage_marker("mm_prefill_enter_model_call", {"use_cache": use_cache}),
        file=sys.stderr,
    )
    with torch.inference_mode():
        prefill = runtime["model"](
            **runtime["inputs"],
            use_cache=use_cache,
            return_dict=True,
        )
    synchronize_npu()
    print(
        format_probe_stage_marker(
            "mm_prefill_done",
            {"has_past_key_values": prefill.past_key_values is not None},
        ),
        file=sys.stderr,
    )
    return {
        "status": "ok",
        "probe_stage": "mm_prefill",
        "prefill_seconds": round(time.perf_counter() - stage_start, 4),
        "prefill_logits_shape": list(prefill.logits.shape),
        "has_past_key_values": prefill.past_key_values is not None,
        "first_kv_debug": extract_first_kv_debug_info(prefill.past_key_values),
    }


def execute_mm_decode1(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    if args.disable_cache:
        raise RuntimeError("mm_decode1 诊断要求 use_cache=True，不能与 --disable-cache 同时使用")
    prefill_start = time.perf_counter()
    print(
        format_probe_stage_marker("mm_decode1_prefill_enter_model_call", {"use_cache": True}),
        file=sys.stderr,
    )
    with torch.inference_mode():
        prefill = runtime["model"](
            **runtime["inputs"],
            use_cache=True,
            return_dict=True,
        )
    synchronize_npu()
    prefill_seconds = round(time.perf_counter() - prefill_start, 4)
    print(
        format_probe_stage_marker(
            "mm_decode1_prefill_done",
            {
                "has_past_key_values": prefill.past_key_values is not None,
                "prefill_logits_shape": list(prefill.logits.shape),
            },
        ),
        file=sys.stderr,
    )

    next_token = prefill.logits[:, -1:].argmax(dim=-1)
    decode_attention_mask = build_decode_attention_mask(runtime["inputs"]["attention_mask"])
    decode_inputs = runtime["model"].prepare_inputs_for_generation(
        next_token,
        past_key_values=prefill.past_key_values,
        attention_mask=decode_attention_mask,
        use_cache=True,
    )
    decode_input_tensor_info = collect_tensor_debug_info(decode_inputs)
    print_tensor_debug_info("decode_inputs", decode_input_tensor_info)

    decode_start = time.perf_counter()
    print(
        format_probe_stage_marker("mm_decode1_decode_enter_model_call", {"use_cache": True}),
        file=sys.stderr,
    )
    with torch.inference_mode():
        decode_output = runtime["model"](
            **decode_inputs,
            use_cache=True,
            return_dict=True,
        )
    synchronize_npu()
    decode_seconds = round(time.perf_counter() - decode_start, 4)
    print(
        format_probe_stage_marker(
            "mm_decode1_decode_done",
            {"decode1_logits_shape": list(decode_output.logits.shape)},
        ),
        file=sys.stderr,
    )

    return {
        "status": "ok",
        "probe_stage": "mm_decode1",
        "prefill_seconds": prefill_seconds,
        "decode1_seconds": decode_seconds,
        "prefill_logits_shape": list(prefill.logits.shape),
        "decode1_logits_shape": list(decode_output.logits.shape),
        "has_past_key_values": prefill.past_key_values is not None,
        "first_kv_debug": extract_first_kv_debug_info(prefill.past_key_values),
        "decode1_first_kv_debug": extract_first_kv_debug_info(decode_output.past_key_values),
        "next_token_ids": next_token.tolist(),
        "decode_attention_mask_shape": list(decode_attention_mask.shape),
        "decode_input_tensor_info": decode_input_tensor_info,
    }


def execute_full_generate(args: argparse.Namespace, runtime: dict[str, Any]) -> dict[str, Any]:
    stage_start = time.perf_counter()
    with torch.inference_mode():
        outputs = runtime["model"].generate(
            **runtime["inputs"],
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    synchronize_npu()
    generated_ids = outputs[:, runtime["prompt_seq_len"]:]
    answer = strip_thinking_markers(
        runtime["tokenizer"].batch_decode(generated_ids, skip_special_tokens=True)[0]
    )
    return {
        "status": "ok",
        "probe_stage": "full_generate",
        "generate_seconds": round(time.perf_counter() - stage_start, 4),
        "first_token_seconds": None,
        "answer": answer,
    }


def run_stage_probe(
    args: argparse.Namespace,
    include_image: bool,
    executor,
) -> int:
    start = time.perf_counter()
    payload = build_base_payload(args)
    runtime: dict[str, Any] | None = None
    try:
        runtime = prepare_runtime_context(args, include_image=include_image)
        add_runtime_payload(payload, runtime)
        reset_npu_peak_stats()
        payload.update(executor(args, runtime))
        payload.update({"total_seconds": round(time.perf_counter() - start, 4)})
        payload.update(collect_runtime_metrics())
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        print(serialized)
        if args.metrics_json:
            Path(args.metrics_json).write_text(serialized)
        return 0
    except BaseException as exc:  # noqa: BLE001
        write_failure_report(args.failure_report, args, exc)
        payload.update(
            {
                "status": "error",
                "failure_type": classify_runtime_issue(exc),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        if runtime is not None:
            add_runtime_payload(payload, runtime)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        print(serialized)
        if args.metrics_json:
            Path(args.metrics_json).write_text(serialized)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


def run_direct_probe(args: argparse.Namespace) -> int:
    if args.probe == "text_prefill":
        return run_stage_probe(args, include_image=False, executor=execute_text_prefill)
    if args.probe == "mm_prefill":
        return run_stage_probe(args, include_image=True, executor=execute_mm_prefill)
    if args.probe == "mm_decode1":
        return run_stage_probe(args, include_image=True, executor=execute_mm_decode1)
    return run_stage_probe(args, include_image=True, executor=execute_full_generate)


def run_probe_parent(args: argparse.Namespace) -> int:
    command = build_child_probe_command(args)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(Path(__file__).resolve().parent),
        env=env,
        start_new_session=True,
    )

    try:
        stdout, stderr = process.communicate(timeout=args.probe_timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        payload = build_base_payload(args)
        payload.update(
            {
                "status": "error",
                "failure_type": "timeout",
                "error_type": "TimeoutExpired",
                "error": f"probe {args.probe} 超时 {args.probe_timeout} 秒",
                "stdout_tail": tail_text(stdout),
                "stderr_tail": tail_text(stderr),
            }
        )
        write_payload_failure_report(args.failure_report, args, payload)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        print(serialized)
        if args.metrics_json:
            Path(args.metrics_json).write_text(serialized)
        return 124

    if stderr:
        print(stderr, file=sys.stderr, end="")

    payload: dict[str, Any] | None = None
    stdout_text = stdout.strip()
    if stdout_text:
        try:
            payload = json.loads(stdout_text)
        except json.JSONDecodeError:
            payload = None

    if payload is None:
        payload = build_base_payload(args)
        payload.update(
            {
                "status": "error",
                "failure_type": "invalid_probe_output",
                "error_type": "JSONDecodeError",
                "error": "子进程未返回合法 JSON",
                "child_returncode": process.returncode,
                "stdout_tail": tail_text(stdout),
                "stderr_tail": tail_text(stderr),
            }
        )
        write_payload_failure_report(args.failure_report, args, payload)
        serialized = json.dumps(payload, ensure_ascii=False, indent=2)
        print(serialized)
        if args.metrics_json:
            Path(args.metrics_json).write_text(serialized)
        return process.returncode or 1

    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    print(serialized)
    if args.metrics_json:
        Path(args.metrics_json).write_text(serialized)
    if payload.get("status") != "ok" and not Path(args.failure_report).exists():
        write_payload_failure_report(args.failure_report, args, payload)
    if process.returncode == 0 and payload.get("status") == "ok":
        return 0
    return process.returncode or 1


def main() -> int:
    args = parse_args()
    if args.child_probe:
        return run_direct_probe(args)
    if args.probe in {"text_prefill", "mm_prefill", "mm_decode1"}:
        return run_probe_parent(args)
    return run_direct_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
