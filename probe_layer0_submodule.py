#!/usr/bin/env python3
"""Qwen3.5-0.8B Layer0 submodule-level hang locator.

Three experiment modes:
  --exp hooks       : forward pre-hook tracing to find last-alive module
  --exp manual      : step-by-step embed_tokens + layer0 isolation
  --exp delta_steps : instrument linear_attn fallback internals stage by stage
  --exp bypass      : replace suspect submodule with passthrough identity
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import inspect
import json
import os
import re
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# ── Utilities ────────────────────────────────────────────────────────────────

def emit(msg: str) -> None:
    t = time.strftime("%H:%M:%S")
    print(f"[{t}] {msg}", file=sys.stderr, flush=True)


def describe_tensor(t: Any) -> str:
    if not isinstance(t, torch.Tensor):
        return f"type={type(t).__name__}"
    return (
        f"shape={tuple(t.shape)} dtype={t.dtype} "
        f"device={t.device} stride={t.stride()} contig={t.is_contiguous()}"
    )


def describe_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return describe_tensor(value)
    if isinstance(value, tuple):
        parts = []
        for item in value[:3]:
            parts.append(describe_tensor(item) if isinstance(item, torch.Tensor) else type(item).__name__)
        return f"tuple({', '.join(parts)})"
    return repr(value)


def describe_callable(fn: Any) -> str:
    if fn is None:
        return "None"
    module_name = getattr(fn, "__module__", "")
    qualname = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None) or type(fn).__name__
    return f"{module_name}.{qualname}" if module_name else str(qualname)


def emit_stage(stage: str, value: Any | None = None, sync: bool = False) -> None:
    if value is None:
        emit(stage)
    else:
        emit(f"{stage}: {describe_value(value)}")
    if sync:
        emit(f"{stage}: synchronize_npu()")
        synchronize_npu()


def synchronize_npu() -> None:
    if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
        torch.npu.synchronize()


def ensure_torch_npu_loaded() -> None:
    if hasattr(torch, "npu"):
        return
    importlib.import_module("torch_npu")


def install_timeout(seconds: int) -> None:
    def handler(signum, frame):
        emit(f"TIMEOUT after {seconds}s — dumping all thread stacks")
        for tid, stack in sys._current_frames().items():
            tname = "unknown"
            for t in threading.enumerate():
                if t.ident == tid:
                    tname = t.name
                    break
            emit(f"--- Thread {tid} ({tname}) ---")
            for line in traceback.format_stack(stack):
                for part in line.rstrip().split("\n"):
                    emit(f"  {part}")
        emit("Exiting with code 124")
        os._exit(124)

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    emit(f"Timeout armed: {seconds}s")


# ── Device & Model Loading ───────────────────────────────────────────────────

def get_device() -> torch.device:
    ensure_torch_npu_loaded()
    if not hasattr(torch, "npu"):
        raise RuntimeError("torch.npu unavailable — torch_npu not loaded")
    if not torch.npu.is_available():
        raise RuntimeError("torch.npu.is_available() is False")
    torch.npu.set_device(0)
    return torch.device("npu:0")


def load_model(model_path: str, device: torch.device):
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    emit(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    config_path = Path(model_path) / "config.json"
    loader = AutoModelForImageTextToText
    if config_path.exists():
        config = json.loads(config_path.read_text())
        archs = config.get("architectures", [])
        if archs:
            emit(f"Architecture: {archs[0]}")
        if archs and "ConditionalGeneration" not in archs[0]:
            from transformers import AutoModelForCausalLM
            loader = AutoModelForCausalLM

    emit(f"Loading model via {loader.__name__}")
    kwargs: dict[str, Any] = {"trust_remote_code": True, "dtype": torch.float16}
    try:
        model = loader.from_pretrained(
            model_path, attn_implementation="eager", **kwargs
        )
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        try:
            model = loader.from_pretrained(
                model_path, attn_implementation="eager", **kwargs
            )
        except TypeError:
            model = loader.from_pretrained(model_path, **kwargs)
    model.eval()

    emit("Moving model to device")
    model.to(device)

    # patch_embed fp16 fix (VL models only)
    try:
        visual = getattr(model, "visual", None) or getattr(
            getattr(model, "model", None), "visual", None
        )
        if visual and hasattr(visual, "patch_embed"):
            proj = visual.patch_embed.proj
            with torch.no_grad():
                proj.weight.data = proj.weight.data.to(dtype=torch.float16)
                if proj.bias is not None:
                    proj.bias.data = proj.bias.data.to(dtype=torch.float16)
            emit("Applied patch_embed fp16 fix")
    except Exception as e:
        emit(f"patch_embed fix skipped: {e}")

    return model, tokenizer


def find_text_model(model):
    """Locate the language model backbone (has embed_tokens + layers)."""
    candidates = [
        "model.language_model",
        "language_model",
        "model.model",
        "model",
    ]
    for path in candidates:
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
            if hasattr(obj, "embed_tokens") and hasattr(obj, "layers"):
                emit(f"Found text_model at: {path}")
                return obj, path
        except AttributeError:
            continue
    return None, None


def tokenize_minimal(tokenizer, device):
    encoded = tokenizer("Hi", return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    emit(f"Tokenized 'Hi': input_ids {describe_tensor(input_ids)}")
    return input_ids, attention_mask


def build_layer0_kwargs(text_model, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> dict[str, Any]:
    batch_size, seq_len = hidden_states.shape[:2]
    cache_position = torch.arange(seq_len, device=hidden_states.device)
    position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
    position_embeddings = text_model.rotary_emb(hidden_states, position_ids)

    layer0 = text_model.layers[0]
    layer_mask = attention_mask
    if getattr(layer0, "layer_type", None) == "linear_attention" and hasattr(text_model, "_update_linear_attn_mask"):
        layer_mask = text_model._update_linear_attn_mask(attention_mask, cache_position)

    return {
        "hidden_states": hidden_states,
        "position_embeddings": position_embeddings,
        "attention_mask": layer_mask,
        "position_ids": position_ids,
        "past_key_values": None,
        "cache_position": cache_position,
    }


def submodule_matches_target(subname: str, target: str, all_layers: bool) -> bool:
    if not subname.startswith("layers."):
        return False

    patterns = {
        "linear_attn": r"^layers\.\d+\.linear_attn$" if all_layers else r"^layers\.0\.linear_attn$",
        "conv1d": r"^layers\.\d+\.linear_attn\.conv1d$" if all_layers else r"^layers\.0\.linear_attn\.conv1d$",
        "mlp": r"^layers\.\d+\.mlp$" if all_layers else r"^layers\.0\.mlp$",
    }
    pattern = patterns.get(target)
    return bool(pattern and re.fullmatch(pattern, subname))


# ── Exp 1: hooks ─────────────────────────────────────────────────────────────

HOOK_TARGETS = [
    "embed_tokens",
    "layers.0",
    "layers.0.input_layernorm",
    "layers.0.linear_attn",
    "layers.0.linear_attn.in_proj_qkv",
    "layers.0.linear_attn.conv1d",
    "layers.0.linear_attn.norm",
    "layers.0.linear_attn.out_proj",
    "layers.0.post_attention_layernorm",
    "layers.0.mlp",
    "layers.1",
]


def run_hooks_experiment(model, tokenizer, device):
    emit("=== Exp 1: forward pre-hook tracing ===")
    input_ids, attention_mask = tokenize_minimal(tokenizer, device)

    text_model, text_model_path = find_text_model(model)
    if text_model is None:
        emit("ERROR: Cannot find text_model for hook registration")
        return

    registered = 0
    for name, module in text_model.named_modules():
        if not name:
            continue
        for target in HOOK_TARGETS:
            if name == target:

                def make_hook(tag_name):
                    def hook_fn(mod, args):
                        first = args[0] if args else None
                        if isinstance(first, torch.Tensor):
                            desc = describe_tensor(first)
                        elif isinstance(first, tuple) and first:
                            desc = f"tuple[0]={describe_tensor(first[0])}"
                        else:
                            desc = f"args_types={[type(a).__name__ for a in args]}"
                        emit(f"PRE-HOOK [{tag_name}] {desc}")
                    return hook_fn

                module.register_forward_pre_hook(make_hook(target))
                full_name = f"{text_model_path}.{name}" if text_model_path else name
                emit(f"  hook: {full_name} -> [{target}]")
                registered += 1
                break

    emit(f"Hooks registered: {registered}")
    emit("Starting forward pass...")
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    synchronize_npu()
    emit(f"Forward completed! logits shape: {tuple(output.logits.shape)}")


# ── Exp 2: manual ────────────────────────────────────────────────────────────

def run_manual_experiment(model, tokenizer, device):
    emit("=== Exp 2: manual step-by-step execution ===")

    text_model, text_model_path = find_text_model(model)
    if text_model is None:
        emit("ERROR: Cannot find text_model with embed_tokens + layers")
        for name, _ in model.named_modules():
            emit(f"  module: {name}")
        return

    input_ids, attention_mask = tokenize_minimal(tokenizer, device)

    # Step 1: embed_tokens
    emit("Step 1: embed_tokens(input_ids)")
    with torch.inference_mode():
        hidden = text_model.embed_tokens(input_ids)
    synchronize_npu()
    emit(f"  result: {describe_tensor(hidden)}")

    # Step 2: inspect layer0
    layer0 = text_model.layers[0]
    emit(f"Step 2: layer0 type = {type(layer0).__name__}")
    try:
        sig = inspect.signature(layer0.forward)
        emit(f"  forward signature: {sig}")
    except (ValueError, TypeError) as e:
        emit(f"  cannot inspect signature: {e}")

    # Step 3: build the same core arguments as text_model.forward for layer0
    emit("Step 3: calling layer0.forward(...) with reconstructed text_model kwargs")
    kwargs = build_layer0_kwargs(text_model, hidden, attention_mask)
    for key, value in kwargs.items():
        if isinstance(value, torch.Tensor):
            emit(f"  kwarg {key}: {describe_tensor(value)}")
        elif isinstance(value, tuple):
            emit(f"  kwarg {key}: tuple({len(value)})")
        else:
            emit(f"  kwarg {key}: {value}")
    try:
        with torch.inference_mode():
            result = layer0(**kwargs)
        synchronize_npu()
        if isinstance(result, torch.Tensor):
            emit(f"  SUCCESS tensor: {describe_tensor(result)}")
        elif isinstance(result, tuple):
            emit(f"  SUCCESS tuple({len(result)})")
        else:
            emit(f"  SUCCESS: {type(result).__name__}")
    except Exception as e:
        emit(f"  {type(e).__name__}: {e}")
        traceback.print_exc(file=sys.stderr)


# ── Exp 3: delta_steps ────────────────────────────────────────────────────────

def trace_torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    l2norm_fn=None,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        if l2norm_fn is None:
            raise ValueError("l2norm_fn is required when use_qk_l2norm_in_kernel=True")
        emit_stage("CHUNK before l2norm")
        query = l2norm_fn(query, dim=-1, eps=1e-6)
        key = l2norm_fn(key, dim=-1, eps=1e-6)
        emit_stage("CHUNK after l2norm_query", query, sync=True)
        emit_stage("CHUNK after l2norm_key", key, sync=True)

    emit_stage("CHUNK before transpose_cast_fp32")
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    emit_stage("CHUNK after fp32_query", query, sync=True)
    emit_stage("CHUNK after fp32_key", key, sync=True)
    emit_stage("CHUNK after fp32_value", value, sync=True)
    emit_stage("CHUNK after fp32_beta", beta, sync=True)
    emit_stage("CHUNK after fp32_g", g, sync=True)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    emit(
        f"CHUNK dims batch={batch_size} heads={num_heads} seq={sequence_length} "
        f"k_head_dim={k_head_dim} v_head_dim={v_head_dim} chunk_size={chunk_size} pad_size={pad_size}"
    )
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    emit_stage("CHUNK after pad_query", query, sync=True)
    emit_stage("CHUNK after pad_key", key, sync=True)
    emit_stage("CHUNK after pad_value", value, sync=True)
    emit_stage("CHUNK after pad_beta", beta, sync=True)
    emit_stage("CHUNK after pad_g", g, sync=True)

    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    emit_stage("CHUNK after scale_query", query, sync=True)

    v_beta = value * beta.unsqueeze(-1)
    emit_stage("CHUNK after v_beta", v_beta, sync=True)
    k_beta = key * beta.unsqueeze(-1)
    emit_stage("CHUNK after k_beta", k_beta, sync=True)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    emit_stage("CHUNK after reshape_query", query, sync=True)
    emit_stage("CHUNK after reshape_key", key, sync=True)
    emit_stage("CHUNK after reshape_value", value, sync=True)
    emit_stage("CHUNK after reshape_k_beta", k_beta, sync=True)
    emit_stage("CHUNK after reshape_v_beta", v_beta, sync=True)

    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    emit_stage("CHUNK after reshape_g", g, sync=True)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    emit_stage("CHUNK after upper_mask", mask)

    g = g.cumsum(dim=-1)
    emit_stage("CHUNK after g_cumsum", g, sync=True)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    emit_stage("CHUNK after decay_mask", decay_mask, sync=True)
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    emit_stage("CHUNK after attn_init", attn, sync=True)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    emit_stage("CHUNK after triangular_update", attn, sync=True)
    emit_stage("CHUNK before attn_eye")
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    emit_stage("CHUNK after attn_eye", attn, sync=True)
    emit_stage("CHUNK before value_update")
    value = attn @ v_beta
    emit_stage("CHUNK after value_update", value, sync=True)
    emit_stage("CHUNK before k_cumdecay")
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    emit_stage("CHUNK after k_cumdecay", k_cumdecay, sync=True)
    emit_stage("CHUNK before init_last_state")
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    emit_stage("CHUNK after init_last_state", last_recurrent_state, sync=True)
    core_attn_out = torch.zeros_like(value)
    emit_stage("CHUNK after init_core_attn_out", core_attn_out, sync=True)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    emit_stage("CHUNK after strict_upper_mask", mask)

    for i in range(0, total_sequence_length // chunk_size):
        emit(f"CHUNK loop i={i} before attn")
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        emit_stage(f"CHUNK loop{i} after attn", attn, sync=True)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        emit_stage(f"CHUNK loop{i} after v_prime", v_prime, sync=True)
        v_new = v_i - v_prime
        emit_stage(f"CHUNK loop{i} after v_new", v_new, sync=True)
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        emit_stage(f"CHUNK loop{i} after attn_inter", attn_inter, sync=True)
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        emit_stage(f"CHUNK loop{i} after core_attn_out_write", core_attn_out[:, :, i], sync=True)
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
        emit_stage(f"CHUNK loop{i} after last_recurrent_state", last_recurrent_state, sync=True)

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    emit_stage("CHUNK after final_reshape", core_attn_out, sync=True)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    emit_stage("CHUNK after trim_sequence", core_attn_out, sync=True)
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    emit_stage("CHUNK after final_transpose_cast", core_attn_out, sync=True)
    return core_attn_out, last_recurrent_state


def run_delta_steps_experiment(model, tokenizer, device):
    emit("=== Exp 3: linear_attn delta-step tracing ===")

    text_model, _ = find_text_model(model)
    if text_model is None:
        emit("ERROR: Cannot find text_model for delta_steps")
        return

    input_ids, attention_mask = tokenize_minimal(tokenizer, device)
    with torch.inference_mode():
        hidden = text_model.embed_tokens(input_ids)
    emit_stage("DELTA after embed_tokens", hidden, sync=True)

    layer0 = text_model.layers[0]
    linear_attn = layer0.linear_attn
    kwargs = build_layer0_kwargs(text_model, hidden, attention_mask)
    emit(f"DELTA layer0 type={type(layer0).__name__} linear_attn type={type(linear_attn).__name__}")
    emit(f"DELTA causal_conv1d_fn={describe_callable(getattr(linear_attn, 'causal_conv1d_fn', None))}")
    emit(f"DELTA chunk_gated_delta_rule={describe_callable(getattr(linear_attn, 'chunk_gated_delta_rule', None))}")

    qwen_module = importlib.import_module(linear_attn.__class__.__module__)
    apply_mask_to_padding_states = getattr(qwen_module, "apply_mask_to_padding_states")
    original_forward = linear_attn.forward

    def instrumented_forward(self, hidden_states, cache_params=None, cache_position=None, attention_mask=None):
        emit_stage("DELTA before apply_mask_to_padding_states")
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        emit_stage("DELTA after apply_mask_to_padding_states", hidden_states, sync=True)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
            cache_params is not None
            and cache_params.has_previous_state
            and seq_len == 1
            and cache_position is not None
        )
        emit(f"DELTA use_precomputed_states={use_precomputed_states}")

        if cache_params is not None:
            conv_state = cache_params.conv_states[self.layer_idx]
            recurrent_state = cache_params.recurrent_states[self.layer_idx]
        else:
            conv_state = None
            recurrent_state = None

        emit_stage("DELTA before in_proj_qkv")
        mixed_qkv = self.in_proj_qkv(hidden_states)
        emit_stage("DELTA after in_proj_qkv", mixed_qkv, sync=True)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        emit_stage("DELTA after transpose_qkv", mixed_qkv, sync=True)

        emit_stage("DELTA before in_proj_z")
        z = self.in_proj_z(hidden_states)
        emit_stage("DELTA after in_proj_z", z, sync=True)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        emit_stage("DELTA after reshape_z", z, sync=True)

        emit_stage("DELTA before in_proj_b")
        b = self.in_proj_b(hidden_states)
        emit_stage("DELTA after in_proj_b", b, sync=True)

        emit_stage("DELTA before in_proj_a")
        a = self.in_proj_a(hidden_states)
        emit_stage("DELTA after in_proj_a", a, sync=True)

        if use_precomputed_states:
            emit(f"DELTA before causal_conv1d_update backend={describe_callable(self.causal_conv1d_update)}")
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
            emit_stage("DELTA after causal_conv1d_update", mixed_qkv, sync=True)
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.conv_states[self.layer_idx] = conv_state
            if self.causal_conv1d_fn is not None:
                emit(f"DELTA before causal_conv1d_fn backend={describe_callable(self.causal_conv1d_fn)}")
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
                emit_stage("DELTA after causal_conv1d_fn", mixed_qkv, sync=True)
            else:
                emit_stage("DELTA before conv1d_fallback")
                conv_raw = self.conv1d(mixed_qkv)
                emit_stage("DELTA after conv1d_raw", conv_raw, sync=True)
                mixed_qkv = F.silu(conv_raw[:, :, :seq_len])
                emit_stage("DELTA after conv1d_fallback", mixed_qkv, sync=True)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        emit_stage("DELTA after transpose_back", mixed_qkv, sync=True)

        emit_stage("DELTA before split_qkv")
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        emit_stage("DELTA after split_query", query, sync=True)
        emit_stage("DELTA after split_key", key, sync=True)
        emit_stage("DELTA after split_value", value, sync=True)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        emit_stage("DELTA after reshape_query", query, sync=True)
        emit_stage("DELTA after reshape_key", key, sync=True)
        emit_stage("DELTA after reshape_value", value, sync=True)

        beta = b.sigmoid()
        emit_stage("DELTA after beta", beta, sync=True)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        emit_stage("DELTA after g", g, sync=True)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            emit_stage("DELTA after repeat_interleave_query", query, sync=True)
            emit_stage("DELTA after repeat_interleave_key", key, sync=True)

        if not use_precomputed_states:
            emit(f"DELTA before chunk_gated_delta_rule backend={describe_callable(self.chunk_gated_delta_rule)}")
            if getattr(self.chunk_gated_delta_rule, "__name__", "") == "torch_chunk_gated_delta_rule":
                core_attn_out, last_recurrent_state = trace_torch_chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=True,
                    l2norm_fn=getattr(qwen_module, "l2norm", None),
                )
            else:
                core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=True,
                )
            emit_stage("DELTA after chunk_gated_delta_rule", core_attn_out, sync=True)
        else:
            emit(f"DELTA before recurrent_gated_delta_rule backend={describe_callable(self.recurrent_gated_delta_rule)}")
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
            emit_stage("DELTA after recurrent_gated_delta_rule", core_attn_out, sync=True)

        if cache_params is not None:
            cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        emit_stage("DELTA before norm", (core_attn_out, z))
        core_attn_out = self.norm(core_attn_out, z)
        emit_stage("DELTA after norm", core_attn_out, sync=True)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        emit_stage("DELTA after reshape_out", core_attn_out, sync=True)

        emit_stage("DELTA before out_proj")
        output = self.out_proj(core_attn_out)
        emit_stage("DELTA after out_proj", output, sync=True)
        return output

    linear_attn.forward = instrumented_forward.__get__(linear_attn, type(linear_attn))
    try:
        with torch.inference_mode():
            result = layer0(**kwargs)
        emit_stage("DELTA layer0 result", result, sync=True)
    finally:
        linear_attn.forward = original_forward


# ── Exp 4: bypass ────────────────────────────────────────────────────────────

class BypassLinearAttn(torch.nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        emit(f"BYPASS linear_attn: {describe_tensor(hidden_states)}")
        return hidden_states


class BypassConv1d(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        emit(f"BYPASS conv1d: {describe_tensor(x)}")
        return x


class BypassMLP(torch.nn.Module):
    def forward(self, x, *args, **kwargs):
        emit(f"BYPASS mlp: {describe_tensor(x)}")
        return x


BYPASS_CLS = {
    "linear_attn": BypassLinearAttn,
    "conv1d": BypassConv1d,
    "mlp": BypassMLP,
}


def run_bypass_experiment(model, tokenizer, device, target: str, all_layers: bool):
    emit(f"=== Exp 3: bypass target={target} all_layers={all_layers} ===")

    bypass_cls = BYPASS_CLS.get(target)
    if bypass_cls is None:
        emit(f"Unknown bypass target: {target}")
        return

    text_model, text_model_path = find_text_model(model)
    if text_model is None:
        emit("ERROR: Cannot find text_model for bypass")
        return

    # Collect replacements first to avoid mutating during iteration
    replacements: list[tuple[str, torch.nn.Module, str]] = []
    for name, _ in text_model.named_modules():
        if not name or not submodule_matches_target(name, target, all_layers):
            continue
        dot = name.rsplit(".", 1)
        parent_path, attr_name = (dot[0], dot[1]) if len(dot) == 2 else ("", name)
        parent = text_model
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        full_name = f"{text_model_path}.{name}" if text_model_path else name
        replacements.append((full_name, parent, attr_name))

    replaced = 0
    for name, parent, attr_name in replacements:
        setattr(parent, attr_name, bypass_cls())
        emit(f"  replaced: {name}")
        replaced += 1

    emit(f"Total replaced: {replaced}")
    if replaced == 0:
        emit("WARNING: No modules matched! Candidates containing target name:")
        for n, _ in text_model.named_modules():
            if target in n:
                emit(f"  {n}")
        return

    input_ids, attention_mask = tokenize_minimal(tokenizer, device)
    emit("Starting forward pass with bypass...")
    with torch.inference_mode():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
    synchronize_npu()
    emit(f"Forward completed! logits shape: {tuple(output.logits.shape)}")


# ── Entry ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Qwen3.5-0.8B Layer0 submodule hang locator"
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--exp", required=True, choices=["hooks", "manual", "delta_steps", "bypass"])
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument(
        "--bypass-target",
        choices=["linear_attn", "conv1d", "mlp"],
        default="linear_attn",
    )
    p.add_argument("--bypass-all-layers", action="store_true")
    return p.parse_args()


def main() -> int:
    faulthandler.enable(file=sys.stderr)
    args = parse_args()
    emit(f"Experiment: {args.exp}")
    emit(f"Model: {args.model_path}")

    device = get_device()
    emit(f"Device: {device}")

    model, tokenizer = load_model(args.model_path, device)
    install_timeout(args.timeout)

    try:
        if args.exp == "hooks":
            run_hooks_experiment(model, tokenizer, device)
        elif args.exp == "manual":
            run_manual_experiment(model, tokenizer, device)
        elif args.exp == "delta_steps":
            run_delta_steps_experiment(model, tokenizer, device)
        elif args.exp == "bypass":
            run_bypass_experiment(
                model, tokenizer, device,
                args.bypass_target, args.bypass_all_layers,
            )
    except Exception:
        emit("EXCEPTION during experiment:")
        traceback.print_exc(file=sys.stderr)
        return 1

    emit("Experiment finished successfully")
    signal.alarm(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
