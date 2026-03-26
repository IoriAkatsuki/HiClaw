import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "edge" / "unified_app" / "control_plane.py"


def load_module():
    if not TARGET.exists():
        raise AssertionError(f"控制平面模块不存在: {TARGET}")
    spec = importlib.util.spec_from_file_location("unified_control_plane_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("unified_control_plane_test", None)
    spec.loader.exec_module(module)
    return module


class UnifiedControlPlaneTest(unittest.TestCase):
    def test_load_runtime_config_returns_defaults(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            config = module.load_runtime_config(root)

            self.assertIn("danger_distance", config)
            self.assertIn("conf_thres", config)
            self.assertIn("camera_serial", config)
            self.assertIn("marker_style", config)
            self.assertEqual(config["marker_style"], "rectangle")

    def test_list_model_candidates_prefers_yolo26m(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a = root / "models"
            b = root / "d435_project" / "projects" / "yolo26_galvo" / "models"
            a.mkdir(parents=True)
            b.mkdir(parents=True)
            (b / "yolo26m_best.om").write_text("m", encoding="utf-8")
            (b / "yolo26n_best.om").write_text("n", encoding="utf-8")
            (a / "notes.txt").write_text("skip", encoding="utf-8")

            models = module.list_model_candidates(root)

            self.assertGreaterEqual(len(models), 2)
            self.assertTrue(models[0]["path"].endswith("yolo26m_best.om"))

    def test_marker_state_round_trip(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            payload = {"selected_track_id": 7, "marker_style": "circle"}

            module.save_marker_state(root, payload)
            actual = module.load_marker_state(root)

            self.assertEqual(actual["selected_track_id"], 7)
            self.assertEqual(actual["marker_style"], "circle")

    def test_solve_homography_returns_matrix(self):
        module = load_module()
        payload = {
            "pixel_points": [[0, 0], [100, 0], [100, 50], [0, 50]],
            "galvo_points": [[10, 20], [110, 20], [110, 70], [10, 70]],
        }

        result = module.solve_homography(payload)

        self.assertEqual(len(result["homography_matrix"]), 3)
        self.assertEqual(len(result["homography_matrix"][0]), 3)
        self.assertLess(result["max_error"], 1e-3)

    def test_apply_calibration_writes_yaml(self):
        module = load_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            calibration_path = root / "edge" / "laser_galvo" / "galvo_calibration.yaml"
            calibration_path.parent.mkdir(parents=True)
            calibration_path.write_text("homography_matrix: [[1,0,0],[0,1,0],[0,0,1]]\n", encoding="utf-8")
            payload = {
                "pixel_points": [[0, 0], [100, 0], [100, 50], [0, 50]],
                "galvo_points": [[10, 20], [110, 20], [110, 70], [10, 70]],
            }

            result = module.apply_calibration(root, str(calibration_path), payload)
            written = calibration_path.read_text(encoding="utf-8")

            self.assertIn("homography_matrix", written)
            self.assertEqual(result["calibration_file"], str(calibration_path))


if __name__ == "__main__":
    unittest.main()
