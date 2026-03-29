#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""激光振镜控制器兼容性测试。"""

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "edge" / "laser_galvo"))

from galvo_controller import LaserGalvoController  # noqa: E402


class DummyLaserGalvoController(LaserGalvoController):
    """测试桩：拦截串口发送命令，避免依赖硬件。"""

    def __init__(self):
        super().__init__(serial_port="/dev/null", baudrate=115200, calibration_file=None)
        self.sent_commands = []
        self.ser = type("DummySerial", (), {"is_open": True, "close": lambda self: None})()

    def _send_text_command(self, command_str):
        self.sent_commands.append(command_str)
        return True


class LaserDrawBoxCompatTest(unittest.TestCase):
    def test_draw_box_supports_legacy_steps_per_edge_arg(self):
        controller = DummyLaserGalvoController()
        ok = controller.draw_box([100, 120, 220, 260], pixel_coords=True, steps_per_edge=15)

        self.assertTrue(ok)
        self.assertEqual(len(controller.sent_commands), 1)
        self.assertTrue(controller.sent_commands[0].startswith("0R,"))

    def test_draw_box_command_is_consistent_with_or_without_steps_arg(self):
        controller_a = DummyLaserGalvoController()
        controller_b = DummyLaserGalvoController()

        controller_a.draw_box([100, 120, 220, 260], pixel_coords=True)
        controller_b.draw_box([100, 120, 220, 260], pixel_coords=True, steps_per_edge=15)

        self.assertEqual(controller_a.sent_commands[0], controller_b.sent_commands[0])

    def test_disconnect_sends_laser_off_command(self):
        controller = DummyLaserGalvoController()

        controller.disconnect()

        self.assertIn("U;", controller.sent_commands)

    def test_update_tasks_uses_semicolon_terminated_u_command(self):
        controller = DummyLaserGalvoController()

        controller.update_tasks()

        self.assertEqual(controller.sent_commands[-1], "U;")

    def test_draw_box_uses_projected_corners_for_size_when_homography_exists(self):
        controller = DummyLaserGalvoController()
        controller.homography_matrix = np.array(
            [[2.0, 0.0, 10.0], [0.0, 3.0, -5.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        controller.draw_box([100, 120, 220, 260], pixel_coords=True)

        self.assertEqual(controller.sent_commands[-1], "0R,330,565,240,420")

    def test_draw_box_sweep_emits_multiple_circle_tasks(self):
        controller = DummyLaserGalvoController()

        ok = controller.draw_box_sweep(
            [100, 120, 220, 260],
            pixel_coords=True,
            image_width=640,
            image_height=480,
            max_tasks=8,
        )

        self.assertTrue(ok)
        self.assertEqual(len(controller.sent_commands), 8)
        self.assertTrue(all(f"{idx}C," in cmd for idx, cmd in enumerate(controller.sent_commands)))


if __name__ == "__main__":
    unittest.main()
