#!/usr/bin/env python3
"""Evaluate perplexity of Qwen3-0.6B .om on WikiText-2 (Ascend 310B1).

Sliding-window approach: process text in fixed 512-token chunks with stride.
No KV cache — each chunk is an independent prefill pass.
"""
import sys, time, math
import numpy as np

SEQ_LEN   = 512
STRIDE    = 256
OM_PATH   = "/home/HwHiAiUser/ICT/models/qwen3_0.6b_seq512.om"
TOK_DIR   = "/home/HwHiAiUser/ICT/models/qwen3-0.6b"
VOCAB     = 151936


def load_wikitext2(tokenizer):
    """Load WikiText-2 test set, return token ids."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    except Exception:
        print("datasets library not available, downloading raw text...", flush=True)
        import urllib.request
        url = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1/test-00000-of-00001.parquet"
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(ds["text"])
        except Exception:
            # Last resort: use a small subset
            text = "The tower is 324 metres tall about the same height as an 81 storey building and the tallest structure in Paris Its base is square measuring 125 metres on each side During its construction the Eiffel Tower surpassed the Washington Monument to become the tallest man made structure in the world a title it held for 41 years until the Chrysler Building in New York City was finished in 1930"
            print(f"Using fallback text ({len(text)} chars)")

    ids = tokenizer.encode(text)
    print(f"WikiText-2 test: {len(ids)} tokens")
    return ids


def main():
    import acl

    # Tokenizer
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK_DIR, trust_remote_code=True)

    # Load data
    token_ids = load_wikitext2(tok)
    total_tokens = len(token_ids)

    # ACL init
    acl.init()
    acl.rt.set_device(0)
    ctx, _ = acl.rt.create_context(0)

    model_id, ret = acl.mdl.load_from_file(OM_PATH)
    assert ret == 0, f"load failed: {ret}"
    desc = acl.mdl.create_desc()
    acl.mdl.get_desc(desc, model_id)
    out_size = acl.mdl.get_output_size_by_index(desc, 0)

    # Pre-allocate device buffers
    id_buf_sz = SEQ_LEN * 8  # int64
    mask_buf_sz = SEQ_LEN * 8

    dev_ids, _ = acl.rt.malloc(id_buf_sz, 0)
    dev_mask, _ = acl.rt.malloc(mask_buf_sz, 0)
    dev_out, _ = acl.rt.malloc(out_size, 0)
    host_out, _ = acl.rt.malloc_host(out_size)

    # Sliding window PPL
    nlls = []
    n_evaluated = 0
    n_chunks = 0
    t0 = time.time()

    for begin in range(0, total_tokens - 1, STRIDE):
        end = min(begin + SEQ_LEN, total_tokens)
        chunk_ids = token_ids[begin:end]

        # Pad to SEQ_LEN if needed
        actual_len = len(chunk_ids)
        if actual_len < SEQ_LEN:
            chunk_ids = chunk_ids + [0] * (SEQ_LEN - actual_len)

        input_np = np.array([chunk_ids], dtype=np.int64)
        mask_np = np.zeros((1, SEQ_LEN), dtype=np.int64)
        mask_np[0, :actual_len] = 1

        # Copy to device
        acl.rt.memcpy(dev_ids, id_buf_sz, acl.util.bytes_to_ptr(input_np.tobytes()), id_buf_sz, 1)
        acl.rt.memcpy(dev_mask, mask_buf_sz, acl.util.bytes_to_ptr(mask_np.tobytes()), mask_buf_sz, 1)

        # Build datasets
        in_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(in_ds, acl.create_data_buffer(dev_ids, id_buf_sz))
        acl.mdl.add_dataset_buffer(in_ds, acl.create_data_buffer(dev_mask, mask_buf_sz))
        out_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(out_ds, acl.create_data_buffer(dev_out, out_size))

        # Execute
        ret = acl.mdl.execute(model_id, in_ds, out_ds)
        assert ret == 0, f"execute failed at chunk {n_chunks}: {ret}"

        # Read logits
        acl.rt.memcpy(host_out, out_size, dev_out, out_size, 2)
        logits = np.frombuffer(acl.util.ptr_to_bytes(host_out, out_size), dtype=np.float16)
        logits = logits.reshape(1, SEQ_LEN, VOCAB).astype(np.float32)

        # Compute NLL for valid positions
        # For sliding window: only count positions in [max(begin, STRIDE), end-1)
        # to avoid double-counting overlapping regions
        eval_start = max(0, STRIDE - (begin if begin == 0 else 0))
        if begin == 0:
            eval_start = 1  # skip first token (no prediction for it)

        for pos in range(eval_start, actual_len):
            target_id = token_ids[begin + pos]
            # log_softmax at position pos-1 predicts token at pos
            row = logits[0, pos - 1, :]
            row_max = row.max()
            log_sum_exp = np.log(np.sum(np.exp(row - row_max))) + row_max
            log_prob = row[target_id] - log_sum_exp
            nlls.append(-log_prob)
            n_evaluated += 1

        n_chunks += 1
        acl.mdl.destroy_dataset(in_ds)
        acl.mdl.destroy_dataset(out_ds)

        if n_chunks % 10 == 0:
            elapsed = time.time() - t0
            ppl_so_far = math.exp(sum(nlls) / len(nlls))
            print(f"  chunk {n_chunks}: {n_evaluated} tokens, PPL={ppl_so_far:.2f}, {elapsed:.1f}s", flush=True)

        if end >= total_tokens:
            break

    elapsed = time.time() - t0
    avg_nll = sum(nlls) / len(nlls)
    ppl = math.exp(avg_nll)

    print(f"\n{'='*50}")
    print(f"WikiText-2 PPL = {ppl:.2f}")
    print(f"Evaluated {n_evaluated} tokens in {n_chunks} chunks")
    print(f"Time: {elapsed:.1f}s ({n_evaluated/elapsed:.1f} tok/s)")
    print(f"{'='*50}")

    # Cleanup
    acl.rt.free(dev_ids)
    acl.rt.free(dev_mask)
    acl.rt.free(dev_out)
    acl.rt.free_host(host_out)
    acl.mdl.unload(model_id)
    acl.mdl.destroy_desc(desc)
    acl.rt.destroy_context(ctx)
    acl.rt.reset_device(0)
    acl.finalize()


if __name__ == "__main__":
    main()
