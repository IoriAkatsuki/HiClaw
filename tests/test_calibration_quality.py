#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动校准核心算法单测（无硬件依赖）。"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'edge' / 'laser_galvo'))

from calibrate_galvo import GalvoCalibrator  # noqa: E402


class CalibrationQualityTest(unittest.TestCase):
    def setUp(self):
        self.calibrator = GalvoCalibrator(
            min_valid_points=7,
            mean_error_thres=300.0,
            max_error_thres=700.0,
        )

    def test_detect_red_laser_spot_with_frame_difference(self):
        frame_off = np.zeros((240, 320, 3), dtype=np.uint8)
        frame_on = frame_off.copy()
        cv2.circle(frame_on, (120, 80), 8, (0, 0, 255), -1)

        det = self.calibrator.detect_laser_spot(frame_on=frame_on, frame_off=frame_off, debug=False)

        self.assertIsNotNone(det)
        x, y = det['pos']
        self.assertLessEqual(abs(x - 120), 3)
        self.assertLessEqual(abs(y - 80), 3)

    def test_robust_aggregate_rejects_outlier(self):
        detections = [
            {'pos': (100.0, 100.0), 'score': 1.0},
            {'pos': (101.0, 99.0), 'score': 1.0},
            {'pos': (99.0, 101.0), 'score': 1.0},
            {'pos': (100.5, 100.2), 'score': 1.0},
            {'pos': (260.0, 260.0), 'score': 1.0},  # 离群点
        ]

        center, stats = self.calibrator._robust_aggregate(detections, min_inliers=4)

        self.assertIsNotNone(center)
        self.assertGreaterEqual(stats['inlier_count'], 4)
        self.assertLess(abs(center[0] - 100.0), 3.0)
        self.assertLess(abs(center[1] - 100.0), 3.0)

    def test_quality_gate_thresholds(self):
        ok, _ = self.calibrator.passes_quality_gate(valid_points=7, mean_error=250.0, max_error=650.0)
        self.assertTrue(ok)

        bad_points, _ = self.calibrator.passes_quality_gate(valid_points=6, mean_error=250.0, max_error=650.0)
        self.assertFalse(bad_points)

        bad_mean, _ = self.calibrator.passes_quality_gate(valid_points=7, mean_error=350.0, max_error=650.0)
        self.assertFalse(bad_mean)

        bad_max, _ = self.calibrator.passes_quality_gate(valid_points=7, mean_error=250.0, max_error=750.0)
        self.assertFalse(bad_max)

    def test_calculate_homography_passes_on_clean_mapping(self):
        c = GalvoCalibrator(
            min_valid_points=7,
            mean_error_thres=300.0,
            max_error_thres=700.0,
        )

        valid_galvo = [
            (-5000, -5000),
            (0, -5000),
            (5000, -5000),
            (-5000, 0),
            (0, 0),
            (5000, 0),
            (-5000, 5000),
            (0, 5000),
            (5000, 5000),
        ]
        pixel_points = []
        for gx, gy in valid_galvo:
            # 构造稳定的仿射映射：pixel -> galvo 可逆
            px = gx / 25.0 + 320.0
            py = gy / 30.0 + 240.0
            pixel_points.append((px, py))

        c.valid_galvo_points = valid_galvo
        c.pixel_points = pixel_points

        self.assertTrue(c.calculate_homography())
        self.assertTrue(c.quality_metrics.get('pass', False))

    def test_calculate_homography_still_uses_pixel_points_when_cam_points_exist(self):
        c = GalvoCalibrator(
            min_valid_points=4,
            mean_error_thres=300.0,
            max_error_thres=700.0,
        )

        c.valid_galvo_points = [
            (-1000, -1000),
            (1000, -1000),
            (1000, 1000),
            (-1000, 1000),
        ]
        c.pixel_points = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
        c.cam_points = [
            (-10.0, -10.0),
            (10.0, -10.0),
            (10.0, 10.0),
            (-10.0, 10.0),
        ]

        self.assertTrue(c.calculate_homography())
        mapped = cv2.perspectiveTransform(
            np.array([[[0.0, 0.0]]], dtype=np.float32),
            c.homography_matrix,
        )
        self.assertAlmostEqual(float(mapped[0][0][0]), -1000.0, places=3)
        self.assertAlmostEqual(float(mapped[0][0][1]), -1000.0, places=3)
        self.assertEqual(c.quality_metrics.get('mapping_source'), 'pixel')

    def test_custom_axis_ranges_expand_grid(self):
        c = GalvoCalibrator(
            grid_size=5,
            x_range=20000,
            y_range=15000,
            grid_margin=5000,
        )

        xs = sorted({point[0] for point in c.galvo_points})
        ys = sorted({point[1] for point in c.galvo_points})

        self.assertEqual(len(c.galvo_points), 25)
        self.assertEqual(xs[0], -20000)
        self.assertEqual(xs[-1], 20000)
        self.assertEqual(ys[0], -15000)
        self.assertEqual(ys[-1], 15000)

    def test_suggest_scan_ranges_expands_fov_but_stays_within_safe_limit(self):
        suggestion = GalvoCalibrator.suggest_scan_ranges(
            galvo_points=[
                (-5000, -5000),
                (5000, -5000),
                (-5000, 5000),
                (5000, 5000),
            ],
            pixel_points=[
                (338.25, 239.25),
                (239.50, 239.25),
                (338.25, 139.17),
                (239.50, 139.17),
            ],
            image_width=640,
            image_height=480,
            safety_factor=0.9,
            grid_margin=5000,
        )

        self.assertGreater(suggestion['x_range'], 5000)
        self.assertGreater(suggestion['y_range'], 5000)
        self.assertLessEqual(suggestion['x_range'], 27767)
        self.assertLessEqual(suggestion['y_range'], 27767)

    def test_show_marker_uses_new_packet_protocol(self):
        class DummyCalibrator(GalvoCalibrator):
            def __init__(self):
                super().__init__()
                self.commands = []

            def _write_command(self, command):
                self.commands.append(command)
                return True

        c = DummyCalibrator()
        ok = c.show_marker(100, -200, radius=300, slot=1)

        self.assertTrue(ok)
        self.assertEqual(c.commands[-1], "1C,100,-200,300;U;")

    def test_clear_tasks_uses_update_only_packet(self):
        class DummyCalibrator(GalvoCalibrator):
            def __init__(self):
                super().__init__()
                self.commands = []

            def _write_command(self, command):
                self.commands.append(command)
                return True

        c = DummyCalibrator()
        ok = c.clear_tasks()

        self.assertTrue(ok)
        self.assertEqual(c.commands[-1], "U;")

    def test_sample_depth_returns_roi_median(self):
        class FakeDepthFrame:
            def __init__(self, mapping):
                self.mapping = mapping

            def get_distance(self, x, y):
                return self.mapping.get((x, y), 0.0)

        c = GalvoCalibrator()
        depth_frame = FakeDepthFrame(
            {
                (10, 10): 0.50,
                (11, 10): 0.52,
                (10, 11): 0.0,
                (11, 11): 0.48,
                (12, 11): 0.51,
            }
        )

        depth = c.sample_depth(depth_frame, 11, 10, radius=1)

        self.assertAlmostEqual(depth, 0.505, places=3)


if __name__ == '__main__':
    unittest.main()
