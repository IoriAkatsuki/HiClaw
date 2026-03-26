#!/usr/bin/env python3
"""Qwen3-0.6B .om single-pass inference test on Ascend 310B1.

This is a prefill-only test (no autoregressive loop, no KV cache).
It runs one forward pass to verify the .om produces valid logits.
"""
import sys, struct, numpy as np
from pathlib import Path

# ── paths ──
OM_PATH  = Path.home() / "ICT/models/qwen3_0.6b_full.om"
TOK_DIR  = Path.home() / "ICT/models/qwen3-0.6b"

def main():
    # ── tokenizer ──
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(str(TOK_DIR), trust_remote_code=True)
    except ImportError:
        print("transformers not installed, using hardcoded token ids")
        tok = None

    prompt = "The capital of France is"
    if tok:
        ids = tok.encode(prompt)
        print(f"Prompt: {prompt!r}  →  token ids: {ids}")
    else:
        ids = [576, 6513, 315, 9822, 374]  # fallback
        print(f"Hardcoded ids: {ids}")

    seq_len = len(ids)

    # ── ACL init ──
    import acl
    ret = acl.init()
    assert ret == 0, f"acl.init failed: {ret}"
    ret = acl.rt.set_device(0)
    assert ret == 0, f"set_device failed: {ret}"
    context, ret = acl.rt.create_context(0)
    assert ret == 0, f"create_context failed: {ret}"

    # ── load model ──
    model_id, ret = acl.mdl.load_from_file(str(OM_PATH))
    assert ret == 0, f"load model failed: {ret}"
    desc = acl.mdl.create_desc()
    ret = acl.mdl.get_desc(desc, model_id)
    assert ret == 0

    n_in  = acl.mdl.get_num_inputs(desc)
    n_out = acl.mdl.get_num_outputs(desc)
    print(f"Model loaded: {n_in} inputs, {n_out} outputs")

    for i in range(n_in):
        dims, ret = acl.mdl.get_input_dims(desc, i)
        name = acl.mdl.get_input_name_by_index(desc, i)
        print(f"  input[{i}] {name}: {dims}")
    for i in range(n_out):
        dims, ret = acl.mdl.get_output_dims(desc, i)
        name = acl.mdl.get_output_name_by_index(desc, i)
        print(f"  output[{i}] {name}: {dims}")

    # ── prepare inputs ──
    input_ids = np.array([ids], dtype=np.int64)          # [1, seq_len]
    attn_mask = np.ones((1, seq_len), dtype=np.int64)    # [1, seq_len]

    dataset = acl.mdl.create_dataset()

    for arr in [input_ids, attn_mask]:
        buf_size = arr.nbytes
        dev_ptr, ret = acl.rt.malloc(buf_size, 0)  # ACL_MEM_MALLOC_NORMAL_ONLY
        assert ret == 0, f"malloc failed: {ret}"
        ret = acl.rt.memcpy(dev_ptr, buf_size, acl.util.numpy_to_ptr(arr), buf_size, 1)  # host→device
        assert ret == 0
        data_buf = acl.create_data_buffer(dev_ptr, buf_size)
        _, ret = acl.mdl.add_dataset_buffer(dataset, data_buf)
        assert ret == 0

    # ── prepare output ──
    out_dataset = acl.mdl.create_dataset()
    out_size = acl.mdl.get_output_size_by_index(desc, 0)
    out_ptr, ret = acl.rt.malloc(out_size, 0)
    assert ret == 0
    out_buf = acl.create_data_buffer(out_ptr, out_size)
    _, ret = acl.mdl.add_dataset_buffer(out_dataset, out_buf)
    assert ret == 0

    # ── execute ──
    print("Running inference...", flush=True)
    ret = acl.mdl.execute(model_id, dataset, out_dataset)
    assert ret == 0, f"execute failed: {ret}"

    # ── read output ──
    host_ptr, ret = acl.rt.malloc_host(out_size)
    assert ret == 0
    ret = acl.rt.memcpy(host_ptr, out_size, out_ptr, out_size, 2)  # device→host
    assert ret == 0

    # Output is logits [1, seq_len, vocab_size] in fp16
    vocab_size = 151936
    logits_flat = np.frombuffer(acl.util.ptr_to_bytes(host_ptr, out_size), dtype=np.float16)
    logits = logits_flat.reshape(1, seq_len, vocab_size)

    # Get top-5 for last token
    last_logits = logits[0, -1, :]
    top5_idx = np.argsort(last_logits)[-5:][::-1]
    print(f"\nTop-5 predictions for next token after '{prompt}':")
    for idx in top5_idx:
        score = float(last_logits[idx])
        token_str = tok.decode([idx]) if tok else f"[{idx}]"
        print(f"  {idx:6d}  {score:8.3f}  {token_str!r}")

    # ── cleanup ──
    acl.mdl.unload(model_id)
    acl.mdl.destroy_desc(desc)
    acl.rt.destroy_context(context)
    acl.rt.reset_device(0)
    acl.finalize()
    print("\nDone.")

if __name__ == "__main__":
    main()
