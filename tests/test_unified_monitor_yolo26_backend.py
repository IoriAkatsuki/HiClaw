import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "edge" / "unified_app" / "unified_monitor.py"


def load_module():
    spec = importlib.util.spec_from_file_location("unified_monitor_yolo26_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("unified_monitor_yolo26_test", None)
    spec.loader.exec_module(module)
    return module


class _FakeBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = np.array([cls_id], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)


class _FakeResult:
    def __init__(self):
        self.names = {0: "fpga", 1: "stm32"}
        self.boxes = [
            _FakeBox(0, 0.91, [10, 20, 110, 220]),
            _FakeBox(1, 0.83, [30, 40, 130, 140]),
        ]


class UnifiedMonitorYolo26BackendTest(unittest.TestCase):
    def test_detect_yolo_backend_uses_file_suffix(self):
        module = load_module()

        self.assertEqual(module.detect_yolo_backend("/tmp/model.om"), "acl")
        self.assertEqual(module.detect_yolo_backend("/tmp/model.pt"), "ultralytics")
        self.assertEqual(module.detect_yolo_backend("/tmp/model.onnx"), "ultralytics")

    def test_convert_ultralytics_result_to_objects(self):
        module = load_module()

        objects = module.convert_ultralytics_result_to_objects(_FakeResult())

        self.assertEqual(len(objects), 2)
        self.assertEqual(objects[0]["name"], "fpga")
        self.assertEqual(objects[0]["cls"], 0)
        self.assertAlmostEqual(objects[0]["score"], 0.91, places=2)
        self.assertEqual(objects[0]["box"], [10.0, 20.0, 110.0, 220.0])
        self.assertEqual(objects[1]["name"], "stm32")

    def test_prepare_acl_yolo_input_rgb_chw_float32(self):
        module = load_module()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[0, 0] = [10, 20, 30]  # BGR

        out = module.prepare_acl_yolo_input(frame, "rgb_chw_float32")

        self.assertEqual(out.shape, (1, 3, 640, 640))
        self.assertEqual(out.dtype, np.float32)
        self.assertAlmostEqual(float(out[0, 0, 0, 0]), 30 / 255.0, places=4)
        self.assertAlmostEqual(float(out[0, 1, 0, 0]), 20 / 255.0, places=4)
        self.assertAlmostEqual(float(out[0, 2, 0, 0]), 10 / 255.0, places=4)


if __name__ == "__main__":
    unittest.main()
