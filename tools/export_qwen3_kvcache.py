#!/usr/bin/env python3
"""Export Qwen3-0.6B as prefill + decode .onnx with explicit KV cache tensors.

Prefill:  input_ids[1,S], attention_mask[1,S] → logits[1,S,V], kv_0_key[1,8,S,128], kv_0_val[1,8,S,128], ...
Decode:   input_ids[1,1], attention_mask[1,P+1], kv_0_key[1,8,P,128], kv_0_val[1,8,P,128], ... → logits[1,1,V], kv_0_key[1,8,P+1,128], ...

Usage:
    python3 tools/export_qwen3_kvcache.py --model-dir models/qwen3-0.6b --seq-len 512 --out-dir local_qwen35_0p8b_onnx
"""
import argparse, os, sys, torch
from pathlib import Path


def make_kv_names(n_layers: int, prefix: str = "past"):
    names = []
    for i in range(n_layers):
        names.append(f"{prefix}.{i}.key")
        names.append(f"{prefix}.{i}.value")
    return names


class PrefillWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         use_cache=True)
        result = [out.logits]
        for layer_kv in out.past_key_values:
            result.append(layer_kv[0])  # key [1, n_kv_heads, seq, head_dim]
            result.append(layer_kv[1])  # value
        return tuple(result)


class DecodeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.config.num_hidden_layers

    def forward(self, input_ids, attention_mask, *past_kv_flat):
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        for i in range(self.n_layers):
            cache.update(past_kv_flat[2 * i], past_kv_flat[2 * i + 1], i)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         past_key_values=cache, use_cache=True)
        result = [out.logits]
        for layer in out.past_key_values.layers:
            result.append(layer.keys)
            result.append(layer.values)
        return tuple(result)


def export_prefill(model, seq_len: int, out_path: str):
    n_layers = model.config.num_hidden_layers
    wrapper = PrefillWrapper(model).eval()

    dummy_ids = torch.randint(0, 1000, (1, seq_len), dtype=torch.long)
    dummy_mask = torch.ones(1, seq_len, dtype=torch.long)

    print(f"  Test forward (prefill, seq={seq_len})...", flush=True)
    with torch.no_grad():
        outs = wrapper(dummy_ids, dummy_mask)
    print(f"  logits: {outs[0].shape}, kv[0]: {outs[1].shape}", flush=True)

    out_names = ["logits"] + make_kv_names(n_layers, "present")

    print(f"  Exporting {out_path}...", flush=True)
    torch.onnx.export(
        wrapper, (dummy_ids, dummy_mask), out_path,
        input_names=["input_ids", "attention_mask"],
        output_names=out_names,
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"  OK: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")


def export_decode(model, past_len: int, out_path: str):
    n_layers = model.config.num_hidden_layers
    n_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    wrapper = DecodeWrapper(model).eval()

    dummy_ids = torch.randint(0, 1000, (1, 1), dtype=torch.long)
    dummy_mask = torch.ones(1, past_len + 1, dtype=torch.long)

    dummy_kv = []
    for _ in range(n_layers):
        k = torch.randn(1, n_kv_heads, past_len, head_dim, dtype=torch.float16)
        v = torch.randn(1, n_kv_heads, past_len, head_dim, dtype=torch.float16)
        dummy_kv.extend([k, v])

    print(f"  Test forward (decode, past={past_len})...", flush=True)
    with torch.no_grad():
        outs = wrapper(dummy_ids, dummy_mask, *dummy_kv)
    print(f"  logits: {outs[0].shape}, kv[0]: {outs[1].shape}", flush=True)

    in_names = ["input_ids", "attention_mask"] + make_kv_names(n_layers, "past")
    out_names = ["logits"] + make_kv_names(n_layers, "present")

    dyn_axes = {"attention_mask": {1: "total_seq"}}
    for i in range(n_layers):
        dyn_axes[f"past.{i}.key"] = {2: "past_seq"}
        dyn_axes[f"past.{i}.value"] = {2: "past_seq"}
        dyn_axes[f"present.{i}.key"] = {2: "total_seq"}
        dyn_axes[f"present.{i}.value"] = {2: "total_seq"}

    print(f"  Exporting {out_path} (dynamic_axes)...", flush=True)
    torch.onnx.export(
        wrapper, (dummy_ids, dummy_mask, *dummy_kv), out_path,
        input_names=in_names,
        output_names=out_names,
        dynamic_axes=dyn_axes,
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"  OK: {os.path.getsize(out_path) / 1024 / 1024:.1f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/qwen3-0.6b")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--out-dir", default="local_qwen35_0p8b_onnx")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM
    print(f"Loading {args.model_dir}...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, dtype=torch.float16, trust_remote_code=True
    )
    model.eval()
    model.config.use_cache = True

    os.makedirs(args.out_dir, exist_ok=True)

    print("\n[1/2] Exporting prefill model...", flush=True)
    export_prefill(model, args.seq_len,
                   os.path.join(args.out_dir, "qwen3_prefill.onnx"))

    print("\n[2/2] Exporting decode model...", flush=True)
    export_decode(model, args.seq_len,
                  os.path.join(args.out_dir, "qwen3_decode.onnx"))

    print("\nDone. Next: ATC convert both .onnx → .om")


if __name__ == "__main__":
    main()
