import unittest

import torch

import repro_npu_triangular_update as repro


class ReproNpuTriangularUpdateTest(unittest.TestCase):
    def test_parse_shape(self):
        self.assertEqual(repro.parse_shape("1,2,3,4,5"), (1, 2, 3, 4, 5))

    def test_triangular_update_preserves_shape_on_cpu(self):
        attn = torch.randn((1, 1, 1, 4, 4), dtype=torch.float32)

        result = repro.triangular_update(attn.clone())

        self.assertEqual(tuple(result.shape), (1, 1, 1, 4, 4))


if __name__ == "__main__":
    unittest.main()
