import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UNIFIED_TARGET = ROOT / "edge" / "unified_app" / "unified_monitor.py"
SERVER_TARGET = ROOT / "edge" / "unified_app" / "webui_server.py"


def load_unified_module():
    spec = importlib.util.spec_from_file_location("unified_monitor_debug_test", UNIFIED_TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("unified_monitor_debug_test", None)
    spec.loader.exec_module(module)
    return module


def load_server_module():
    spec = importlib.util.spec_from_file_location("webui_server_debug_test", SERVER_TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("webui_server_debug_test", None)
    spec.loader.exec_module(module)
    return module


class UnifiedMonitorDebugOverlayTest(unittest.TestCase):
    def test_build_debug_overlay_returns_rgba_image(self):
        module = load_unified_module()
        overlay = module.build_debug_overlay(
            320,
            240,
            [
                {
                    "box": [10.0, 20.0, 110.0, 120.0],
                    "name": "fpga",
                    "score": 0.91,
                    "track_id": 3,
                }
            ],
            highlighted_track_ids={3},
        )

        self.assertEqual(overlay.shape, (240, 320, 4))
        self.assertEqual(overlay.dtype, np.uint8)
        self.assertGreater(int(overlay[..., 3].max()), 0)

    def test_sync_template_assets_copies_debug_html(self):
        module = load_server_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir)
            module.sync_template_assets(target_dir)

            self.assertTrue((target_dir / "index.html").exists())
            self.assertTrue((target_dir / "debug.html").exists())


if __name__ == "__main__":
    unittest.main()
