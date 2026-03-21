"""ACL backend: .om model inference on Ascend NPU (prefill-only and KV cache modes)."""
import numpy as np
from pathlib import Path

_acl = None
N_LAYERS = 28
N_KV_HEADS = 8
HEAD_DIM = 128
KV_BYTES_PER_POS = N_KV_HEADS * HEAD_DIM * 2  # fp16


def _import_acl():
    global _acl
    if _acl is None:
        import acl as _acl_mod
        _acl = _acl_mod
    return _acl


class OmModel:
    """Thin wrapper around an .om model loaded via ACL (prefill-only, no KV cache)."""

    def __init__(self, om_path: str, device_id: int = 0):
        acl = _import_acl()
        acl.init()
        acl.rt.set_device(device_id)
        self._ctx, _ = acl.rt.create_context(device_id)
        self._device_id = device_id

        self._model_id, ret = acl.mdl.load_from_file(om_path)
        assert ret == 0, f"load_from_file failed: {ret}"
        self._desc = acl.mdl.create_desc()
        acl.mdl.get_desc(self._desc, self._model_id)

        dims, _ = acl.mdl.get_input_dims(self._desc, 0)
        self.seq_len = dims["dims"][1]
        self._out_size = acl.mdl.get_output_size_by_index(self._desc, 0)

        self._id_sz = self.seq_len * 8
        self._dev_ids, _ = acl.rt.malloc(self._id_sz, 0)
        self._dev_mask, _ = acl.rt.malloc(self._id_sz, 0)
        self._dev_out, _ = acl.rt.malloc(self._out_size, 0)
        self._host_out, _ = acl.rt.malloc_host(self._out_size)

    def forward(self, token_ids: list[int]) -> np.ndarray:
        acl = _import_acl()
        seq = list(token_ids)
        actual_len = len(seq)
        if actual_len > self.seq_len:
            seq = seq[-self.seq_len:]
            actual_len = self.seq_len
        elif actual_len < self.seq_len:
            seq = seq + [0] * (self.seq_len - actual_len)

        ids_np = np.array([seq], dtype=np.int64)
        mask_np = np.zeros((1, self.seq_len), dtype=np.int64)
        mask_np[0, :actual_len] = 1

        acl.rt.memcpy(self._dev_ids, self._id_sz,
                       acl.util.bytes_to_ptr(ids_np.tobytes()), self._id_sz, 1)
        acl.rt.memcpy(self._dev_mask, self._id_sz,
                       acl.util.bytes_to_ptr(mask_np.tobytes()), self._id_sz, 1)

        in_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(in_ds, acl.create_data_buffer(self._dev_ids, self._id_sz))
        acl.mdl.add_dataset_buffer(in_ds, acl.create_data_buffer(self._dev_mask, self._id_sz))
        out_ds = acl.mdl.create_dataset()
        acl.mdl.add_dataset_buffer(out_ds, acl.create_data_buffer(self._dev_out, self._out_size))

        ret = acl.mdl.execute(self._model_id, in_ds, out_ds)
        acl.mdl.destroy_dataset(in_ds)
        acl.mdl.destroy_dataset(out_ds)
        if ret != 0:
            raise RuntimeError(f"acl.mdl.execute failed: {ret}")

        acl.rt.memcpy(self._host_out, self._out_size, self._dev_out, self._out_size, 2)
        flat = np.frombuffer(acl.util.ptr_to_bytes(self._host_out, self._out_size), dtype=np.float16)
        vocab = flat.size // self.seq_len
        return flat.reshape(self.seq_len, vocab).astype(np.float32)[:actual_len]

    def close(self):
        acl = _import_acl()
        acl.rt.free(self._dev_ids)
        acl.rt.free(self._dev_mask)
        acl.rt.free(self._dev_out)
        acl.rt.free_host(self._host_out)
        acl.mdl.unload(self._model_id)
        acl.mdl.destroy_desc(self._desc)
        acl.rt.destroy_context(self._ctx)
        acl.rt.reset_device(self._device_id)
        acl.finalize()


# ── helpers for KV cache model ──

def _load_om(path: str):
    acl = _import_acl()
    mid, ret = acl.mdl.load_from_file(path)
    assert ret == 0, f"load failed: {ret}"
    desc = acl.mdl.create_desc()
    acl.mdl.get_desc(desc, mid)
    return mid, desc


def _make_dataset(bufs):
    acl = _import_acl()
    ds = acl.mdl.create_dataset()
    for ptr, sz in bufs:
        acl.mdl.add_dataset_buffer(ds, acl.create_data_buffer(ptr, sz))
    return ds


def _h2d(host_np, dev_ptr, sz):
    acl = _import_acl()
    acl.rt.memcpy(dev_ptr, sz, acl.util.bytes_to_ptr(host_np.tobytes()), sz, 1)


def _d2h(dev_ptr, host_ptr, sz):
    acl = _import_acl()
    acl.rt.memcpy(host_ptr, sz, dev_ptr, sz, 2)


class KVCacheModel:
    """Dual-model backend: prefill + decode with explicit KV cache on device."""

    def __init__(self, prefill_om: str, decode_om: str, device_id: int = 0):
        acl = _import_acl()
        acl.init()
        acl.rt.set_device(device_id)
        self._ctx, _ = acl.rt.create_context(device_id)
        self._device_id = device_id

        self._pf_mid, self._pf_desc = _load_om(prefill_om)
        self._dc_mid, self._dc_desc = _load_om(decode_om)

        # Get seq_len from prefill input[0]
        dims, _ = acl.mdl.get_input_dims(self._pf_desc, 0)
        self.max_seq = dims["dims"][1]  # 512

        # Get vocab from prefill output[0] size
        pf_out0_sz = acl.mdl.get_output_size_by_index(self._pf_desc, 0)
        self.vocab = pf_out0_sz // (self.max_seq * 2)  # fp16

        # KV cache size per layer: [1, N_KV_HEADS, max_seq, HEAD_DIM] fp16
        self._kv_layer_sz = N_KV_HEADS * self.max_seq * HEAD_DIM * 2

        # Allocate device KV cache buffers (stay on device between steps)
        self._dev_kv = []  # list of (key_ptr, val_ptr) per layer
        for _ in range(N_LAYERS):
            kp, _ = acl.rt.malloc(self._kv_layer_sz, 0)
            vp, _ = acl.rt.malloc(self._kv_layer_sz, 0)
            self._dev_kv.append((kp, vp))

        # Prefill buffers
        self._pf_id_sz = self.max_seq * 8
        self._pf_dev_ids, _ = acl.rt.malloc(self._pf_id_sz, 0)
        self._pf_dev_mask, _ = acl.rt.malloc(self._pf_id_sz, 0)

        # Prefill output: logits + 56 KV tensors
        self._pf_dev_logits, _ = acl.rt.malloc(pf_out0_sz, 0)
        self._pf_logits_sz = pf_out0_sz
        # KV outputs point to the same _dev_kv buffers (share device mem)
        self._pf_dev_kv_out = []
        for _ in range(N_LAYERS):
            kp, _ = acl.rt.malloc(self._kv_layer_sz, 0)
            vp, _ = acl.rt.malloc(self._kv_layer_sz, 0)
            self._pf_dev_kv_out.append((kp, vp))

        # Decode buffers
        self._dc_id_sz = 1 * 8  # [1,1] int64
        self._dc_mask_sz = (self.max_seq + 1) * 8  # [1, max_seq+1]
        self._dc_dev_ids, _ = acl.rt.malloc(self._dc_id_sz, 0)
        self._dc_dev_mask, _ = acl.rt.malloc(self._dc_mask_sz, 0)
        dc_logits_sz = acl.mdl.get_output_size_by_index(self._dc_desc, 0)
        self._dc_dev_logits, _ = acl.rt.malloc(dc_logits_sz, 0)
        self._dc_logits_sz = dc_logits_sz

        # Decode KV output buffers (past_len+1 sized, reuse same as input for next step)
        self._dc_dev_kv_out = []
        for _ in range(N_LAYERS):
            kp, _ = acl.rt.malloc(self._kv_layer_sz + N_KV_HEADS * HEAD_DIM * 2, 0)
            vp, _ = acl.rt.malloc(self._kv_layer_sz + N_KV_HEADS * HEAD_DIM * 2, 0)
            self._dc_dev_kv_out.append((kp, vp))

        # Host logits buffer (only need last token)
        self._host_logits, _ = acl.rt.malloc_host(max(pf_out0_sz, dc_logits_sz))

        self._kv_pos = 0  # current KV cache fill level

    def prefill(self, token_ids: list[int]) -> np.ndarray:
        """Run prefill. Returns last-token logits [vocab]. Populates KV cache on device."""
        acl = _import_acl()
        seq = list(token_ids)
        actual_len = len(seq)
        if actual_len > self.max_seq:
            seq = seq[-self.max_seq:]
            actual_len = self.max_seq
        if actual_len < self.max_seq:
            seq = seq + [0] * (self.max_seq - actual_len)

        ids_np = np.array([seq], dtype=np.int64)
        mask_np = np.zeros((1, self.max_seq), dtype=np.int64)
        mask_np[0, :actual_len] = 1

        _h2d(ids_np, self._pf_dev_ids, self._pf_id_sz)
        _h2d(mask_np, self._pf_dev_mask, self._pf_id_sz)

        # Build input dataset
        in_bufs = [(self._pf_dev_ids, self._pf_id_sz),
                    (self._pf_dev_mask, self._pf_id_sz)]
        in_ds = _make_dataset(in_bufs)

        # Build output dataset: logits + 56 KV
        out_bufs = [(self._pf_dev_logits, self._pf_logits_sz)]
        for kp, vp in self._pf_dev_kv_out:
            out_bufs.append((kp, self._kv_layer_sz))
            out_bufs.append((vp, self._kv_layer_sz))
        out_ds = _make_dataset(out_bufs)

        ret = acl.mdl.execute(self._pf_mid, in_ds, out_ds)
        acl.mdl.destroy_dataset(in_ds)
        acl.mdl.destroy_dataset(out_ds)
        if ret != 0:
            raise RuntimeError(f"prefill execute failed: {ret}")

        # Copy KV outputs to the main KV cache buffers
        for i in range(N_LAYERS):
            acl.rt.memcpy(self._dev_kv[i][0], self._kv_layer_sz,
                          self._pf_dev_kv_out[i][0], self._kv_layer_sz, 4)  # d2d
            acl.rt.memcpy(self._dev_kv[i][1], self._kv_layer_sz,
                          self._pf_dev_kv_out[i][1], self._kv_layer_sz, 4)

        self._kv_pos = actual_len

        # Read logits for last token
        _d2h(self._pf_dev_logits, self._host_logits, self._pf_logits_sz)
        flat = np.frombuffer(acl.util.ptr_to_bytes(self._host_logits, self._pf_logits_sz),
                             dtype=np.float16)
        logits = flat.reshape(self.max_seq, self.vocab).astype(np.float32)
        return logits[actual_len - 1]

    def decode_step(self, token_id: int) -> np.ndarray:
        """Run one decode step. Returns logits [vocab]. Updates KV cache."""
        acl = _import_acl()

        ids_np = np.array([[token_id]], dtype=np.int64)
        mask_np = np.zeros((1, self.max_seq + 1), dtype=np.int64)
        mask_np[0, :self._kv_pos + 1] = 1

        _h2d(ids_np, self._dc_dev_ids, self._dc_id_sz)
        _h2d(mask_np, self._dc_dev_mask, self._dc_mask_sz)

        # Input: ids, mask, 56 × KV cache
        in_bufs = [(self._dc_dev_ids, self._dc_id_sz),
                    (self._dc_dev_mask, self._dc_mask_sz)]
        for kp, vp in self._dev_kv:
            in_bufs.append((kp, self._kv_layer_sz))
            in_bufs.append((vp, self._kv_layer_sz))
        in_ds = _make_dataset(in_bufs)

        # Output: logits + 56 × updated KV (past_len+1)
        new_kv_sz = self._kv_layer_sz + N_KV_HEADS * HEAD_DIM * 2
        out_bufs = [(self._dc_dev_logits, self._dc_logits_sz)]
        for kp, vp in self._dc_dev_kv_out:
            out_bufs.append((kp, new_kv_sz))
            out_bufs.append((vp, new_kv_sz))
        out_ds = _make_dataset(out_bufs)

        ret = acl.mdl.execute(self._dc_mid, in_ds, out_ds)
        acl.mdl.destroy_dataset(in_ds)
        acl.mdl.destroy_dataset(out_ds)
        if ret != 0:
            raise RuntimeError(f"decode execute failed: {ret}")

        # Copy updated KV back (new size = kv_pos+1)
        for i in range(N_LAYERS):
            acl.rt.memcpy(self._dev_kv[i][0], new_kv_sz,
                          self._dc_dev_kv_out[i][0], new_kv_sz, 4)
            acl.rt.memcpy(self._dev_kv[i][1], new_kv_sz,
                          self._dc_dev_kv_out[i][1], new_kv_sz, 4)

        self._kv_pos += 1

        _d2h(self._dc_dev_logits, self._host_logits, self._dc_logits_sz)
        flat = np.frombuffer(acl.util.ptr_to_bytes(self._host_logits, self._dc_logits_sz),
                             dtype=np.float16)
        return flat.reshape(self.vocab).astype(np.float32)

    def reset_kv(self):
        self._kv_pos = 0

    def close(self):
        acl = _import_acl()
        acl.mdl.unload(self._pf_mid)
        acl.mdl.unload(self._dc_mid)
        acl.mdl.destroy_desc(self._pf_desc)
        acl.mdl.destroy_desc(self._dc_desc)
        # Free all device buffers (simplified — in production track all ptrs)
        acl.rt.destroy_context(self._ctx)
        acl.rt.reset_device(self._device_id)
        acl.finalize()
