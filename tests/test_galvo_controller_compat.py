#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""激光振镜控制器兼容性测试。"""

import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "edge" / "laser_galvo"))

from galvo_controller import LaserGalvoController  # noqa: E402


class DummyLaserGalvoController(LaserGalvoController):
    """测试桩：拦截串口发送命令，避免依赖硬件。"""

    def __init__(self):
        super().__init__(serial_port="/dev/null", baudrate=115200, calibration_file=None)
        self.sent_commands = []

    def _send_text_command(self, command_str):
        self.sent_commands.append(command_str)
        return True


class LaserDrawBoxCompatTest(unittest.TestCase):
    def test_draw_box_supports_legacy_steps_per_edge_arg(self):
        controller = DummyLaserGalvoController()
        ok = controller.draw_box([100, 120, 220, 260], pixel_coords=True, steps_per_edge=15)

        self.assertTrue(ok)
        self.assertEqual(len(controller.sent_commands), 1)
        self.assertTrue(controller.sent_commands[0].startswith("R0"))

    def test_draw_box_command_is_consistent_with_or_without_steps_arg(self):
        controller_a = DummyLaserGalvoController()
        controller_b = DummyLaserGalvoController()

        controller_a.draw_box([100, 120, 220, 260], pixel_coords=True)
        controller_b.draw_box([100, 120, 220, 260], pixel_coords=True, steps_per_edge=15)

        self.assertEqual(controller_a.sent_commands[0], controller_b.sent_commands[0])


if __name__ == "__main__":
    unittest.main()
