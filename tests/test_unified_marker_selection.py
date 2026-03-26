import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "edge" / "unified_app" / "unified_monitor_mp.py"


def load_module():
    spec = importlib.util.spec_from_file_location("unified_marker_selection_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("unified_marker_selection_test", None)
    spec.loader.exec_module(module)
    return module


class DummyGalvo:
    def __init__(self):
        self.calls = []

    def draw_box(self, box, **kwargs):
        self.calls.append(("box", box, kwargs))
        return True

    def pixel_to_galvo(self, x, y, image_width, image_height):
        return (int(x * 10), int(y * 10))

    def draw_circle(self, x, y, radius, task_index=None):
        self.calls.append(("circle", (x, y, radius, task_index), {}))
        return True


class UnifiedMarkerSelectionTest(unittest.TestCase):
    def test_emit_marker_command_uses_rectangle(self):
        module = load_module()
        galvo = DummyGalvo()

        module.emit_marker_command(galvo, "rectangle", [10.0, 20.0, 30.0, 50.0], frame_width=640, frame_height=480)

        self.assertEqual(galvo.calls[0][0], "box")

    def test_emit_marker_command_uses_circle(self):
        module = load_module()
        galvo = DummyGalvo()

        module.emit_marker_command(galvo, "circle", [10.0, 20.0, 30.0, 50.0], frame_width=640, frame_height=480)

        self.assertEqual(galvo.calls[0][0], "circle")
        _, payload, _ = galvo.calls[0]
        self.assertGreater(payload[2], 0)


if __name__ == "__main__":
    unittest.main()
