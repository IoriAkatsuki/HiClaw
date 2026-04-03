import importlib.util
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MONITOR_TARGET = ROOT / "edge" / "unified_app" / "unified_monitor_mp.py"
INDEX_TARGET = ROOT / "webui_http_unified" / "index.html"


def load_monitor_module():
    spec = importlib.util.spec_from_file_location("unified_monitor_mp_test", MONITOR_TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("unified_monitor_mp_test", None)
    spec.loader.exec_module(module)
    return module


class WebUiStreamDefaultsTest(unittest.TestCase):
    def test_writer_fps_default_is_25(self):
        module = load_monitor_module()

        parser = module.build_arg_parser()
        args = parser.parse_args([
            "--yolo-model", "dummy.om",
            "--data-yaml", "dummy.yaml",
        ])

        self.assertEqual(args.writer_fps, 25.0)

    def test_frame_polling_interval_matches_25fps_fallback(self):
        html = INDEX_TARGET.read_text(encoding="utf-8")

        self.assertRegex(
            html,
            re.compile(r"DOM\.frame\.src = `frame\.jpg\?t=\$\{Date\.now\(\)\}`;\s*}, 40\);"),
        )

    def test_depth_pipeline_defaults_off_when_hand_and_distance_disabled(self):
        module = load_monitor_module()

        parser = module.build_arg_parser()
        args = parser.parse_args([
            "--yolo-model", "dummy.om",
            "--data-yaml", "dummy.yaml",
            "--disable-hand",
            "--disable-distance",
        ])

        self.assertFalse(module.should_enable_depth_pipeline(args))

    def test_h264_stream_path_disabled_by_default(self):
        module = load_monitor_module()

        self.assertFalse(module.h264_stream_enabled())


if __name__ == "__main__":
    unittest.main()
