#!/usr/bin/env python3
"""Test dynamic-shape decode .om with KV cache on 310B1."""
import numpy as np
import time

PREFILL_OM = "/home/HwHiAiUser/ICT/models/qwen3_prefill.om"
DECODE_OM  = "/home/HwHiAiUser/ICT/models/qwen3_decode_dyn.om"
TOK_DIR    = "/home/HwHiAiUser/ICT/models/qwen3-0.6b"

N_LAYERS = 28
N_KV_HEADS = 8
HEAD_DIM = 128
VOCAB = 151936
MAX_SEQ = 512


def main():
    import acl
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)

    acl.init()
    acl.rt.set_device(0)
    ctx, _ = acl.rt.create_context(0)

    # Load prefill model (fixed shape)
    pf_mid, ret = acl.mdl.load_from_file(PREFILL_OM)
    assert ret == 0
    pf_desc = acl.mdl.create_desc()
    acl.mdl.get_desc(pf_desc, pf_mid)

    # Load decode model (dynamic shape)
    dc_mid, ret = acl.mdl.load_from_file(DECODE_OM)
    assert ret == 0
    dc_desc = acl.mdl.create_desc()
    acl.mdl.get_desc(dc_desc, dc_mid)

    print(f"Prefill inputs: {acl.mdl.get_num_inputs(pf_desc)}, outputs: {acl.mdl.get_num_outputs(pf_desc)}")
    print(f"Decode  inputs: {acl.mdl.get_num_inputs(dc_desc)}, outputs: {acl.mdl.get_num_outputs(dc_desc)}")

    # ── Prefill ──
    prompt = "1+1="
    ids = tok.encode(prompt)
    seq_len = len(ids)
    padded_ids = ids + [0] * (MAX_SEQ - seq_len)

    ids_np = np.array([padded_ids], dtype=np.int64)
    mask_np = np.zeros((1, MAX_SEQ), dtype=np.int64)
    mask_np[0, :seq_len] = 1

    # Allocate and copy inputs
    def alloc_and_copy(arr):
        sz = arr.nbytes
        ptr, _ = acl.rt.malloc(sz, 0)
        acl.rt.memcpy(ptr, sz, acl.util.bytes_to_ptr(arr.tobytes()), sz, 1)
        return ptr, sz

    pf_ids_ptr, pf_ids_sz = alloc_and_copy(ids_np)
    pf_mask_ptr, pf_mask_sz = alloc_and_copy(mask_np)

    pf_in = acl.mdl.create_dataset()
    acl.mdl.add_dataset_buffer(pf_in, acl.create_data_buffer(pf_ids_ptr, pf_ids_sz))
    acl.mdl.add_dataset_buffer(pf_in, acl.create_data_buffer(pf_mask_ptr, pf_mask_sz))

    # Prefill outputs: logits + 56 KV tensors
    kv_layer_sz = N_KV_HEADS * MAX_SEQ * HEAD_DIM * 2  # fp16
    logits_sz = MAX_SEQ * VOCAB * 2

    pf_out = acl.mdl.create_dataset()
    pf_logits_ptr, _ = acl.rt.malloc(logits_sz, 0)
    acl.mdl.add_dataset_buffer(pf_out, acl.create_data_buffer(pf_logits_ptr, logits_sz))

    kv_ptrs = []  # list of (key_ptr, val_ptr) per layer
    for _ in range(N_LAYERS):
        kp, _ = acl.rt.malloc(kv_layer_sz, 0)
        vp, _ = acl.rt.malloc(kv_layer_sz, 0)
        kv_ptrs.append((kp, vp))
        acl.mdl.add_dataset_buffer(pf_out, acl.create_data_buffer(kp, kv_layer_sz))
        acl.mdl.add_dataset_buffer(pf_out, acl.create_data_buffer(vp, kv_layer_sz))

    print(f"Running prefill ({seq_len} tokens)...", flush=True)
    t0 = time.time()
    ret = acl.mdl.execute(pf_mid, pf_in, pf_out)
    assert ret == 0, f"prefill failed: {ret}"
    pf_time = time.time() - t0
    print(f"Prefill done: {pf_time*1000:.0f}ms")

    # Read prefill logits, get first token
    host_logits, _ = acl.rt.malloc_host(logits_sz)
    acl.rt.memcpy(host_logits, logits_sz, pf_logits_ptr, logits_sz, 2)
    logits_all = np.frombuffer(acl.util.ptr_to_bytes(host_logits, logits_sz), dtype=np.float16)
    logits_all = logits_all.reshape(MAX_SEQ, VOCAB).astype(np.float32)
    first_logits = logits_all[seq_len - 1]
    first_tok = int(np.argmax(first_logits))
    print(f"First token: {first_tok} = {tok.decode([first_tok])!r} (score={first_logits[first_tok]:.3f})")

    # ── Decode loop ──
    # KV from prefill: [1, 8, 512, 128] but only first seq_len positions are valid
    # For decode: pass KV sliced to actual past_len
    generated = [first_tok]
    past_len = seq_len  # KV has this many valid positions

    # Allocate decode output buffers
    dc_logits_sz = 1 * VOCAB * 2  # [1, 1, V] fp16
    dc_logits_ptr, _ = acl.rt.malloc(dc_logits_sz, 0)
    dc_host_logits, _ = acl.rt.malloc_host(dc_logits_sz)

    # Decode KV output: [1, 8, past_len+1, 128]
    # Max size: [1, 8, 513, 128]
    max_kv_out_sz = N_KV_HEADS * (MAX_SEQ + 1) * HEAD_DIM * 2
    dc_kv_out_ptrs = []
    for _ in range(N_LAYERS):
        kp, _ = acl.rt.malloc(max_kv_out_sz, 0)
        vp, _ = acl.rt.malloc(max_kv_out_sz, 0)
        dc_kv_out_ptrs.append((kp, vp))

    print(f"\nDecoding (greedy, max 20 tokens)...", flush=True)
    t_dec_start = time.time()

    for step in range(20):
        new_tok = generated[-1]
        total_len = past_len + 1  # attention covers past + new token

        # Input: ids[1,1], mask[1, total_len], kv[1, 8, past_len, 128] × 56
        dc_ids_np = np.array([[new_tok]], dtype=np.int64)
        dc_mask_np = np.ones((1, total_len), dtype=np.int64)

        dc_ids_ptr, dc_ids_sz = alloc_and_copy(dc_ids_np)
        dc_mask_ptr, dc_mask_sz = alloc_and_copy(dc_mask_np)

        dc_in = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(dc_in, acl.create_data_buffer(dc_ids_ptr, dc_ids_sz))
        acl.mdl.add_dataset_buffer(dc_in, acl.create_data_buffer(dc_mask_ptr, dc_mask_sz))

        # KV input: slice to actual past_len
        actual_kv_sz = N_KV_HEADS * past_len * HEAD_DIM * 2
        for kp, vp in kv_ptrs:
            acl.mdl.add_dataset_buffer(dc_in, acl.create_data_buffer(kp, actual_kv_sz))
            acl.mdl.add_dataset_buffer(dc_in, acl.create_data_buffer(vp, actual_kv_sz))

        # Output: logits + kv[1, 8, past_len+1, 128] × 56
        new_kv_sz = N_KV_HEADS * (past_len + 1) * HEAD_DIM * 2
        dc_out = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(dc_out, acl.create_data_buffer(dc_logits_ptr, dc_logits_sz))
        for kp, vp in dc_kv_out_ptrs:
            acl.mdl.add_dataset_buffer(dc_out, acl.create_data_buffer(kp, new_kv_sz))
            acl.mdl.add_dataset_buffer(dc_out, acl.create_data_buffer(vp, new_kv_sz))

        ret = acl.mdl.execute(dc_mid, dc_in, dc_out)
        acl.mdl.destroy_dataset(dc_in)
        acl.mdl.destroy_dataset(dc_out)
        acl.rt.free(dc_ids_ptr)
        acl.rt.free(dc_mask_ptr)

        if ret != 0:
            print(f"Decode step {step} failed: {ret}")
            break

        # Copy output KV back to input KV buffers
        for i in range(N_LAYERS):
            acl.rt.memcpy(kv_ptrs[i][0], new_kv_sz, dc_kv_out_ptrs[i][0], new_kv_sz, 4)
            acl.rt.memcpy(kv_ptrs[i][1], new_kv_sz, dc_kv_out_ptrs[i][1], new_kv_sz, 4)

        past_len += 1

        # Read logits
        acl.rt.memcpy(dc_host_logits, dc_logits_sz, dc_logits_ptr, dc_logits_sz, 2)
        step_logits = np.frombuffer(acl.util.ptr_to_bytes(dc_host_logits, dc_logits_sz), dtype=np.float16)
        step_logits = step_logits.astype(np.float32)
        next_tok = int(np.argmax(step_logits))

        if next_tok in (151643, 151645):  # EOS
            print(f"  EOS at step {step}")
            break
        generated.append(next_tok)

    dec_time = time.time() - t_dec_start
    n_dec = len(generated) - 1  # exclude first prefill token
    print(f"\nDecode: {n_dec} tokens in {dec_time:.2f}s ({n_dec/dec_time:.1f} tok/s)")
    print(f"Generated: {tok.decode(generated, skip_special_tokens=True)}")

    acl.mdl.unload(pf_mid)
    acl.mdl.unload(dc_mid)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()


if __name__ == "__main__":
    main()
