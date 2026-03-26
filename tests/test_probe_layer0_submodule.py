import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import probe_layer0_submodule as probe


class ProbeLayer0SubmoduleTest(unittest.TestCase):
    def test_ensure_torch_npu_loaded_imports_module_when_missing(self):
        fake_torch = SimpleNamespace()

        with mock.patch.object(probe, "torch", fake_torch), \
             mock.patch("probe_layer0_submodule.importlib.import_module") as import_module:
            probe.ensure_torch_npu_loaded()

        import_module.assert_called_once_with("torch_npu")

    def test_bypass_linear_attn_returns_tensor(self):
        hidden = torch.zeros((1, 2, 3), dtype=torch.float16)

        output = probe.BypassLinearAttn()(hidden)

        self.assertIs(output, hidden)

    def test_submodule_matches_target_avoids_mtp_and_non_exact_suffix_hits(self):
        self.assertTrue(probe.submodule_matches_target("layers.0.linear_attn", "linear_attn", all_layers=False))
        self.assertTrue(probe.submodule_matches_target("layers.7.linear_attn.conv1d", "conv1d", all_layers=True))
        self.assertFalse(probe.submodule_matches_target("mtp.layers.0.mlp", "mlp", all_layers=False))
        self.assertFalse(probe.submodule_matches_target("layers.0.linear_attn.in_proj_qkv", "linear_attn", all_layers=False))

    def test_build_layer0_kwargs_uses_position_embeddings_for_linear_attention(self):
        captured = {}

        class DummyTextModel:
            def __init__(self):
                self.layers = [SimpleNamespace(layer_type="linear_attention")]

            def rotary_emb(self, hidden_states, position_ids):
                captured["hidden_shape"] = tuple(hidden_states.shape)
                captured["position_ids_shape"] = tuple(position_ids.shape)
                return ("cos", "sin")

            def _update_linear_attn_mask(self, attention_mask, cache_position):
                captured["cache_position"] = cache_position.clone()
                return None

        hidden = torch.zeros((1, 2, 4), dtype=torch.float16)
        attention_mask = torch.ones((1, 2), dtype=torch.int64)

        kwargs = probe.build_layer0_kwargs(DummyTextModel(), hidden, attention_mask)

        self.assertEqual(kwargs["position_embeddings"], ("cos", "sin"))
        self.assertEqual(tuple(kwargs["position_ids"].shape), (3, 1, 2))
        self.assertEqual(tuple(kwargs["cache_position"].shape), (2,))
        self.assertIsNone(kwargs["attention_mask"])
        self.assertEqual(captured["hidden_shape"], (1, 2, 4))
        self.assertEqual(captured["position_ids_shape"], (3, 1, 2))

    def test_parse_args_accepts_delta_steps(self):
        with mock.patch("sys.argv", [
            "probe_layer0_submodule.py",
            "--model-path", "/tmp/model",
            "--exp", "delta_steps",
        ]):
            args = probe.parse_args()

        self.assertEqual(args.exp, "delta_steps")

    def test_trace_torch_chunk_gated_delta_rule_returns_expected_shape(self):
        query = torch.ones((1, 1, 1, 2), dtype=torch.float32)
        key = torch.ones((1, 1, 1, 2), dtype=torch.float32)
        value = torch.ones((1, 1, 1, 2), dtype=torch.float32)
        g = torch.zeros((1, 1, 1), dtype=torch.float32)
        beta = torch.ones((1, 1, 1), dtype=torch.float32)

        output, last_state = probe.trace_torch_chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            chunk_size=4,
            use_qk_l2norm_in_kernel=False,
        )

        self.assertEqual(tuple(output.shape), (1, 1, 1, 2))
        self.assertIsNone(last_state)


if __name__ == "__main__":
    unittest.main()
