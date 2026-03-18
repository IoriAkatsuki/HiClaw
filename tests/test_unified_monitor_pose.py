import importlib.util
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / 'edge' / 'unified_app' / 'unified_monitor.py'


def load_module():
    spec = importlib.util.spec_from_file_location('unified_monitor_test', TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop('unified_monitor_test', None)
    spec.loader.exec_module(module)
    return module


class _DepthFrame:
    def __init__(self, mapping):
        self.mapping = mapping

    def get_distance(self, x, y):
        return self.mapping.get((x, y), 0.0)


class UnifiedMonitorPoseTest(unittest.TestCase):
    def test_build_arg_parser_supports_pose_model(self):
        module = load_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(['--yolo-model', 'det.om', '--pose-model', 'pose.om', '--data-yaml', 'cfg.yaml'])

        self.assertEqual(args.pose_model, 'pose.om')
        self.assertEqual(args.pose_conf_thres, 0.5)
        self.assertEqual(args.pose_infer_skip, 1)

    def test_postprocess_pose_decodes_single_person(self):
        module = load_module()
        pred = [320.0, 240.0, 100.0, 200.0, 2.0] + [0.0] * 51
        pred[5 + 9 * 3: 5 + 9 * 3 + 3] = [320.0, 240.0, 0.9]
        pred[5 + 10 * 3: 5 + 10 * 3 + 3] = [160.0, 120.0, 0.8]
        flat = module.np.array(pred, dtype=module.np.float32)

        persons = module.postprocess_pose([flat], (480, 640, 3), conf_thres=0.5)

        self.assertEqual(len(persons), 1)
        self.assertAlmostEqual(persons[0]['box'][0], 270.0, places=3)
        self.assertAlmostEqual(persons[0]['box'][1], 105.0, places=3)
        self.assertEqual(len(persons[0]['keypoints']), 17)

    def test_extract_pose_wrists_returns_depth_values(self):
        module = load_module()
        person = {
            'keypoints': [[0.0, 0.0, 0.0] for _ in range(17)]
        }
        person['keypoints'][9] = [100.0, 120.0, 0.7]
        person['keypoints'][10] = [300.0, 220.0, 0.8]
        depth_frame = _DepthFrame({(100, 120): 0.25, (300, 220): 0.4})

        wrists = module.extract_pose_wrists(person, depth_frame, conf_thres=0.5)

        self.assertEqual(len(wrists), 2)
        self.assertEqual(wrists[0]['pixel'], (100, 120))
        self.assertAlmostEqual(wrists[0]['depth_mm'], 250.0)
        self.assertEqual(wrists[1]['pixel'], (300, 220))
        self.assertAlmostEqual(wrists[1]['depth_mm'], 400.0)

    def test_prepare_io_buffers_accepts_tuple_status_from_acl_dataset_add(self):
        module = load_module()

        class _FakeRt:
            @staticmethod
            def malloc(size, flag):
                return object(), 0

            @staticmethod
            def malloc_host(size):
                return object(), 0

        class _FakeMdl:
            @staticmethod
            def create_dataset():
                return []

            @staticmethod
            def add_dataset_buffer(dataset, data_buffer):
                dataset.append(data_buffer)
                return dataset, 0

        fake_acl = types.SimpleNamespace(
            rt=_FakeRt(),
            mdl=_FakeMdl(),
            create_data_buffer=lambda buf, size: (buf, size),
        )

        module.acl = fake_acl
        model = module.AclLiteModel('dummy.om')
        model.input_sizes = [4]
        model.output_sizes = [4]

        model._prepare_io_buffers()

        self.assertEqual(len(model.input_data_buffers), 1)
        self.assertEqual(len(model.output_data_buffers), 1)
        self.assertEqual(len(model.host_output_buffers), 1)


if __name__ == '__main__':
    unittest.main()
