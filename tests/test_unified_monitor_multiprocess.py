import importlib.util
import multiprocessing as mp
import queue
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "edge" / "unified_app" / "unified_monitor_mp.py"


def load_module():
    spec = importlib.util.spec_from_file_location("unified_monitor_mp_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("unified_monitor_mp_test", None)
    spec.loader.exec_module(module)
    return module


class UnifiedMonitorMultiprocessTest(unittest.TestCase):
    def test_serialize_objects_for_state_marks_locked_objects(self):
        module = load_module()
        details, laser_objects = module.serialize_objects_for_state(
            [
                {"track_id": 3, "name": "fpga", "score": 0.91234, "box": [10.2, 20.6, 110.1, 220.9]},
                {"track_id": 7, "name": "stm32", "score": 0.83456, "box": [30.2, 40.4, 130.5, 140.7]},
            ],
            locked_track_ids={7},
        )

        self.assertEqual(len(details), 2)
        self.assertFalse(details[0]["locked"])
        self.assertTrue(details[1]["locked"])
        self.assertEqual(details[0]["box"], [10, 21, 110, 221])
        self.assertEqual(details[0]["score"], 0.9123)
        self.assertEqual(laser_objects[1]["track_id"], 7)

    def test_plan_worker_cpus_prefers_distinct_cores(self):
        module = load_module()
        plan = module.plan_worker_cpus(
            ["camera_capture", "detector", "writer", "laser_worker"],
            available_cpus=[0, 1, 2, 3],
        )

        self.assertEqual(
            plan,
            {
                "camera_capture": 0,
                "detector": 1,
                "writer": 2,
                "laser_worker": 3,
            },
        )

    def test_put_latest_message_discards_stale_payload(self):
        module = load_module()
        latest_queue = queue.Queue(maxsize=1)

        self.assertTrue(module.put_latest_message(latest_queue, {"seq": 1}))
        self.assertTrue(module.put_latest_message(latest_queue, {"seq": 2}))
        self.assertEqual(latest_queue.qsize(), 1)
        self.assertEqual(latest_queue.get_nowait(), {"seq": 2})

    def test_shared_json_slot_round_trip(self):
        module = load_module()
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
        slot = module.create_text_slot(ctx, 256)

        module.write_json_slot(slot, {"locked_track_ids": [2, 5], "laser_active": True})

        self.assertEqual(
            module.read_json_slot(slot, default={}),
            {"locked_track_ids": [2, 5], "laser_active": True},
        )


if __name__ == "__main__":
    unittest.main()
