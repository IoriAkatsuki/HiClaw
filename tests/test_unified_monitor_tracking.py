import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "edge" / "unified_app" / "unified_monitor.py"


def load_module():
    spec = importlib.util.spec_from_file_location("unified_monitor_tracking_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("unified_monitor_tracking_test", None)
    spec.loader.exec_module(module)
    return module


class UnifiedMonitorTrackingTest(unittest.TestCase):
    def test_expand_box_scales_about_center(self):
        module = load_module()

        expanded = module.expand_box([100.0, 120.0, 200.0, 220.0], 1.2)

        self.assertEqual(expanded, [90.0, 110.0, 210.0, 230.0])

    def test_smooth_box_blends_previous_and_current_box(self):
        module = load_module()

        smoothed = module.smooth_box(
            [100.0, 100.0, 200.0, 200.0],
            [120.0, 80.0, 220.0, 180.0],
            alpha=0.25,
        )

        self.assertEqual(smoothed, [105.0, 95.0, 205.0, 195.0])

    def test_should_refresh_laser_box_ignores_small_jitter(self):
        module = load_module()

        changed = module.should_refresh_laser_box(
            [100.0, 100.0, 200.0, 200.0],
            [103.0, 98.0, 203.0, 198.0],
            center_threshold_px=8.0,
            size_threshold_ratio=0.12,
        )

        self.assertFalse(changed)

    def test_should_refresh_laser_box_detects_large_motion(self):
        module = load_module()

        changed = module.should_refresh_laser_box(
            [100.0, 100.0, 200.0, 200.0],
            [130.0, 100.0, 230.0, 200.0],
            center_threshold_px=8.0,
            size_threshold_ratio=0.12,
        )

        self.assertTrue(changed)

    def test_assign_track_ids_reuses_id_for_nearby_detection(self):
        module = load_module()
        tracks = {}
        next_track_id = 1

        objects1 = [
            {"box": [10.0, 10.0, 50.0, 50.0], "score": 0.92, "cls": 0, "name": "chip"}
        ]
        tracked1, tracks, next_track_id = module.assign_track_ids(
            objects1, tracks, next_track_id, iou_thres=0.3, max_missing=2
        )

        self.assertEqual(tracked1[0]["track_id"], 1)
        self.assertEqual(next_track_id, 2)

        objects2 = [
            {"box": [12.0, 12.0, 52.0, 52.0], "score": 0.88, "cls": 0, "name": "chip"}
        ]
        tracked2, tracks, next_track_id = module.assign_track_ids(
            objects2, tracks, next_track_id, iou_thres=0.3, max_missing=2
        )

        self.assertEqual(tracked2[0]["track_id"], 1)
        self.assertEqual(next_track_id, 2)

    def test_select_laser_targets_prefers_previously_locked_track(self):
        module = load_module()
        objects = [
            {"box": [10.0, 10.0, 60.0, 60.0], "score": 0.75, "cls": 0, "name": "chip", "track_id": 3},
            {"box": [100.0, 100.0, 180.0, 180.0], "score": 0.99, "cls": 1, "name": "board", "track_id": 7},
        ]

        targets = module.select_laser_targets(
            objects,
            min_score=0.7,
            max_targets=1,
            allowed_classes=None,
            preferred_track_ids={3},
        )

        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["track_id"], 3)

    def test_assign_track_ids_handles_matched_and_new_objects_in_same_frame(self):
        module = load_module()
        tracks = {
            4: {"box": [10.0, 10.0, 50.0, 50.0], "name": "chip", "cls": 0, "score": 0.9, "missing": 0}
        }

        tracked, tracks, next_track_id = module.assign_track_ids(
            [
                {"box": [12.0, 12.0, 52.0, 52.0], "score": 0.91, "cls": 0, "name": "chip"},
                {"box": [100.0, 100.0, 160.0, 180.0], "score": 0.87, "cls": 1, "name": "board"},
            ],
            tracks,
            5,
            iou_thres=0.3,
            max_missing=2,
        )

        self.assertEqual(tracked[0]["track_id"], 4)
        self.assertEqual(tracked[1]["track_id"], 5)
        self.assertEqual(next_track_id, 6)


if __name__ == "__main__":
    unittest.main()
