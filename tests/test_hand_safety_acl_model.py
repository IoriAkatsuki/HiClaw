import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "edge" / "route_b_app" / "hand_safety_monitor.py"


def load_module(fake_acl):
    spec = importlib.util.spec_from_file_location("hand_safety_monitor_acl_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    previous_acl = sys.modules.get("acl")
    sys.modules["acl"] = fake_acl
    sys.modules.pop("hand_safety_monitor_acl_test", None)
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_acl is None:
            sys.modules.pop("acl", None)
        else:
            sys.modules["acl"] = previous_acl
    return module


class HandSafetyAclModelTest(unittest.TestCase):
    def test_prepare_io_buffers_builds_reusable_datasets_and_host_buffers(self):
        counters = {"create_dataset": 0, "malloc_host": 0}

        class _FakeRt:
            @staticmethod
            def malloc(size, flag):
                return object(), 0

            @staticmethod
            def malloc_host(size):
                counters["malloc_host"] += 1
                return object(), 0

            @staticmethod
            def memcpy(dst, dst_size, src, src_size, kind):
                return 0

            @staticmethod
            def set_context(_context):
                return 0

        class _FakeMdl:
            @staticmethod
            def create_dataset():
                counters["create_dataset"] += 1
                return []

            @staticmethod
            def add_dataset_buffer(dataset, data_buffer):
                dataset.append(data_buffer)
                return 0

            @staticmethod
            def execute(model_id, input_dataset, output_dataset):
                return 0

        fake_acl = types.SimpleNamespace(
            rt=_FakeRt(),
            mdl=_FakeMdl(),
            create_data_buffer=lambda buf, size: (buf, size),
            util=types.SimpleNamespace(
                numpy_to_ptr=lambda arr: arr,
                ptr_to_numpy=lambda ptr, shape, dtype: np.zeros(shape, dtype=np.float32),
            ),
        )

        module = load_module(fake_acl)
        model = module.AclLiteModel("pose.om")
        model.context = object()
        model.model_id = 1
        model.input_sizes = [4]
        model.output_sizes = [8]

        model._prepare_io_buffers()

        self.assertEqual(counters["create_dataset"], 2)
        self.assertEqual(counters["malloc_host"], 1)
        self.assertEqual(len(model.input_data_buffers), 1)
        self.assertEqual(len(model.output_data_buffers), 1)
        self.assertEqual(len(model.host_output_buffers), 1)

        image = np.zeros((4,), dtype=np.uint8)
        outputs = model.execute(image)

        self.assertEqual(counters["create_dataset"], 2)
        self.assertEqual(counters["malloc_host"], 1)
        self.assertEqual(len(outputs), 1)


if __name__ == "__main__":
    unittest.main()
