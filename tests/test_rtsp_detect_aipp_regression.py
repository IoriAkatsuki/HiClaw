import importlib.util
import inspect
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "edge" / "route_a_app" / "rtsp_detect_aipp.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("rtsp_detect_aipp_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    fake_acl = types.SimpleNamespace()
    fake_cv2 = types.SimpleNamespace()
    with mock.patch.dict(sys.modules, {"acl": fake_acl, "cv2": fake_cv2}, clear=False):
        spec.loader.exec_module(module)
    return module


class RtspDetectAippRegressionTest(unittest.TestCase):
    def test_postprocess_defaults_keep_conservative_semantics(self):
        module = _load_module()
        sig = inspect.signature(module.postprocess)

        self.assertEqual(sig.parameters["conf_thres"].default, 0.15)
        self.assertFalse(sig.parameters["apply_sigmoid"].default)

    def test_source_uses_wall_clock_timestamp(self):
        source = TARGET.read_text(encoding="utf-8")

        self.assertIn('"ts": time.time()', source)

    def test_source_no_longer_sets_auto_exposure_property(self):
        source = TARGET.read_text(encoding="utf-8")

        self.assertNotIn("CAP_PROP_AUTO_EXPOSURE", source)
