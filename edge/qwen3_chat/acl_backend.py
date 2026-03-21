"""ACL backend: load .om model and run single-pass inference on Ascend NPU."""
import numpy as np

_acl = None

def _import_acl():
    global _acl
    if _acl is None:
        import acl as _acl_mod
        _acl = _acl_mod
    return _acl


class OmModel:
    """Thin wrapper around an .om model loaded via ACL."""

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

        # Pre-allocate device buffers
        self._id_sz = self.seq_len * 8
        self._dev_ids, _ = acl.rt.malloc(self._id_sz, 0)
        self._dev_mask, _ = acl.rt.malloc(self._id_sz, 0)
        self._dev_out, _ = acl.rt.malloc(self._out_size, 0)
        self._host_out, _ = acl.rt.malloc_host(self._out_size)

    def forward(self, token_ids: list[int]) -> np.ndarray:
        """Run one prefill pass. Returns logits [seq_len, vocab_size] as float32."""
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

        acl.rt.memcpy(self._host_out, self._out_size,
                       self._dev_out, self._out_size, 2)

        flat = np.frombuffer(
            acl.util.ptr_to_bytes(self._host_out, self._out_size), dtype=np.float16
        )
        vocab = flat.size // self.seq_len
        logits = flat.reshape(self.seq_len, vocab).astype(np.float32)
        return logits[:actual_len]

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
