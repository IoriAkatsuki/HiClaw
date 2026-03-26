import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MONITOR_TARGET = ROOT / "edge" / "unified_app" / "unified_monitor_mp.py"
CONTROL_TARGET = ROOT / "edge" / "unified_app" / "control_plane.py"
START_TARGET = ROOT / "start_unified.sh"
INDEX_TARGET = ROOT / "webui_http_unified" / "index.html"


def load_module(module_name: str, target: Path):
    spec = importlib.util.spec_from_file_location(module_name, target)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop(module_name, None)
    spec.loader.exec_module(module)
    return module


class WriterRefreshAndDetectionTogglesTest(unittest.TestCase):
    def test_plan_writer_publish_keeps_stream_running_without_new_detection(self):
        module = load_module("writer_refresh_plan_test", MONITOR_TARGET)

        plan = module.plan_writer_publish(
            now=1.0,
            next_write_time=0.5,
            current_detect_seq=7,
            last_snapshot_detect_seq=7,
        )

        self.assertTrue(plan["emit_frame"])
        self.assertFalse(plan["refresh_snapshot"])

    def test_render_live_frame_from_snapshot_draws_cached_annotations(self):
        module = load_module("writer_refresh_render_test", MONITOR_TARGET)

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        snapshot = {
            "object_details": [
                {"box": [10, 10, 30, 30], "name": "fpga", "score": 0.91, "track_id": 3, "locked": True}
            ],
            "is_danger": True,
            "danger_object": "fpga",
        }

        rendered = module.render_live_frame_from_snapshot(frame, snapshot)

        self.assertEqual(rendered.shape, frame.shape)
        self.assertGreater(int(rendered.sum()), 0)

    def test_build_arg_parser_accepts_disable_distance_flag(self):
        module = load_module("writer_refresh_parser_test", MONITOR_TARGET)

        parser = module.build_arg_parser()
        args = parser.parse_args([
            "--yolo-model", "dummy.om",
            "--data-yaml", "dummy.yaml",
            "--disable-distance",
        ])

        self.assertTrue(args.disable_distance)

    def test_control_plane_defaults_disable_hand_and_distance_detection(self):
        module = load_module("control_plane_toggle_defaults_test", CONTROL_TARGET)
        with tempfile.TemporaryDirectory() as tmpdir:
            config = module.default_runtime_config(Path(tmpdir))

        self.assertFalse(config["enable_hand_detection"])
        self.assertFalse(config["enable_distance_detection"])

    def test_shell_env_exports_include_detection_toggle_flags(self):
        module = load_module("control_plane_toggle_exports_test", CONTROL_TARGET)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            module.save_runtime_config(
                root,
                {
                    "enable_hand_detection": False,
                    "enable_distance_detection": False,
                },
            )
            exports = module.shell_env_exports(root)

        self.assertIn("CONTROL_ENABLE_HAND_DETECTION=0", exports)
        self.assertIn("CONTROL_ENABLE_DISTANCE_DETECTION=0", exports)

    def test_start_script_wires_detection_toggle_envs_and_flags(self):
        script = START_TARGET.read_text(encoding="utf-8")

        self.assertIn("CONTROL_ENABLE_HAND_DETECTION", script)
        self.assertIn("CONTROL_ENABLE_DISTANCE_DETECTION", script)
        self.assertIn("--disable-hand", script)
        self.assertIn("--disable-distance", script)

    def test_start_scripts_bootstrap_user_site_packages_for_background_launch(self):
        start_script = START_TARGET.read_text(encoding="utf-8")
        laser_script = (ROOT / "edge" / "unified_app" / "start_unified_with_laser.sh").read_text(encoding="utf-8")

        self.assertIn("site.getusersitepackages()", start_script)
        self.assertIn("site.getusersitepackages()", laser_script)
        self.assertIn("site.getsitepackages()", start_script)
        self.assertIn("site.getsitepackages()", laser_script)
        self.assertIn("export PYTHONPATH", start_script)
        self.assertIn("export PYTHONPATH", laser_script)

    def test_webui_config_exposes_detection_toggle_switches(self):
        html = INDEX_TARGET.read_text(encoding="utf-8")

        self.assertIn('id="config-enable-hand-detection"', html)
        self.assertIn('id="config-enable-distance-detection"', html)
        self.assertRegex(
            html,
            re.compile(r"enable_hand_detection:\s*Boolean\(DOM\.configEnableHandDetection\.checked\)"),
        )
        self.assertRegex(
            html,
            re.compile(r"enable_distance_detection:\s*Boolean\(DOM\.configEnableDistanceDetection\.checked\)"),
        )


if __name__ == "__main__":
    unittest.main()
