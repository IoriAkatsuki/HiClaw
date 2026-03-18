import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "Detect_choose_serial.py"


def load_module():
    spec = importlib.util.spec_from_file_location("detect_choose_serial_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DetectChooseSerialMappingTest(unittest.TestCase):
    def test_load_calibration_homography_reads_yaml_matrix(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.yaml"
            yaml.safe_dump(
                {"homography_matrix": [[2.0, 0.0, 10.0], [0.0, 3.0, -5.0], [0.0, 0.0, 1.0]]},
                path.open("w", encoding="utf-8"),
                allow_unicode=True,
            )

            homography = module.load_calibration_homography(str(path))

        self.assertIsInstance(homography, np.ndarray)
        self.assertEqual(homography.shape, (3, 3))
        self.assertAlmostEqual(float(homography[0, 0]), 2.0)

    def test_px2galvo_prefers_homography_when_provided(self):
        module = load_module()
        homography = np.array([[2.0, 0.0, 10.0], [0.0, 3.0, -5.0], [0.0, 0.0, 1.0]], dtype=np.float32)

        gx, gy = module.px2galvo(100, 50, homography=homography)

        self.assertEqual(gx, 210)
        self.assertEqual(gy, 145)


if __name__ == "__main__":
    unittest.main()
