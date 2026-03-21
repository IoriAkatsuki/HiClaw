import importlib.util
import multiprocessing as mp
import queue
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "edge" / "unified_app" / "unified_monitor_mp.py"
START_SCRIPT = ROOT / "start_unified.sh"
LASER_RUN_SCRIPT = ROOT / "edge" / "laser_galvo" / "run.sh"
LASER_START_SCRIPT = ROOT / "edge" / "unified_app" / "start_unified_with_laser.sh"
BOARD_MODEL_DIR = "/home/HwHiAiUser/ICT/d435_project/projects/yolo26_galvo/models"


def load_module():
    spec = importlib.util.spec_from_file_location("unified_monitor_mp_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("unified_monitor_mp_test", None)
    spec.loader.exec_module(module)
    return module


class UnifiedMonitorMultiprocessTest(unittest.TestCase):
    def test_build_arg_parser_disables_debug_assets_by_default(self):
        module = load_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(["--yolo-model", "det.om", "--data-yaml", "cfg.yaml"])

        self.assertFalse(args.write_debug_assets)
        self.assertEqual(args.writer_fps, 10.0)
        self.assertEqual(args.camera_serial, "")

    def test_build_webui_artifact_plan_skips_debug_assets_by_default(self):
        module = load_module()

        plan = module.build_webui_artifact_plan(write_debug_assets=False)

        self.assertEqual(plan, ("frame.jpg", "state.json"))

    def test_runtime_shared_state_initializes_homography_slot(self):
        module = load_module()
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())

        runtime = module.create_runtime_shared_state(ctx, laser_enabled=False)

        self.assertIsNone(module.read_json_slot(runtime.homography_matrix, default="missing"))

    def test_writer_worker_accepts_write_debug_assets_flag(self):
        module = load_module()

        class _DummyShm:
            def close(self):
                return None

        def _attach(_name, shape, dtype):
            return _DummyShm(), np.zeros(shape, dtype=dtype)

        original_attach = module.attach_shared_buffer
        original_home = module.Path.home
        module.attach_shared_buffer = _attach
        tmpdir = tempfile.TemporaryDirectory()
        module.Path.home = classmethod(lambda cls: Path(tmpdir.name))
        try:
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
            stop_event = ctx.Event()
            stop_event.set()
            runtime = module.create_runtime_shared_state(ctx, laser_enabled=False)

            detect_ready_evt = ctx.Event()
            detect_ready_evt.set()
            module.writer_worker(
                "color",
                "annotated",
                "overlay",
                ctx.Value("i", 0),
                ctx.Value("i", 1),
                detect_ready_evt,
                runtime,
                ["fpga"],
                queue.Queue(maxsize=1),
                stop_event,
                writer_fps=10.0,
                worker_cpu=None,
                write_debug_assets=False,
                config_snapshot={
                    "danger_distance": 300,
                    "conf_thres": 0.55,
                    "yolo_model": "det.om",
                },
            )
        finally:
            module.attach_shared_buffer = original_attach
            module.Path.home = original_home
            tmpdir.cleanup()

    def test_writer_worker_writes_config_snapshot_without_outer_args(self):
        module = load_module()

        class _DummyShm:
            def close(self):
                return None

        def _attach(_name, shape, dtype):
            return _DummyShm(), np.zeros(shape, dtype=dtype)

        original_attach = module.attach_shared_buffer
        original_home = module.Path.home
        original_imwrite = module.cv2.imwrite
        original_dump = module.json.dump
        module.attach_shared_buffer = _attach
        module.cv2.imwrite = lambda *_args, **_kwargs: True
        tmpdir = tempfile.TemporaryDirectory()
        module.Path.home = classmethod(lambda cls: Path(tmpdir.name))
        try:
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
            stop_event = ctx.Event()
            runtime = module.create_runtime_shared_state(ctx, laser_enabled=False)
            module.write_json_slot(runtime.homography_matrix, [[1.0, 0.0, 5.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])

            writer_queue = queue.Queue(maxsize=1)
            writer_queue.put({"fps": 12.5, "yolo_ms": 8.0, "hand_ms": 4.0, "objects": 1, "object_details": []})

            def _dump(payload, handle, *args, **kwargs):
                original_dump(payload, handle, *args, **kwargs)
                stop_event.set()

            module.json.dump = _dump

            detect_ready_evt = ctx.Event()
            detect_ready_evt.set()
            module.writer_worker(
                "color",
                "annotated",
                "overlay",
                ctx.Value("i", 7),
                ctx.Value("i", 1),
                detect_ready_evt,
                runtime,
                ["fpga"],
                writer_queue,
                stop_event,
                writer_fps=10.0,
                worker_cpu=None,
                write_debug_assets=False,
                config_snapshot={
                    "danger_distance": 300,
                    "conf_thres": 0.55,
                    "yolo_model": "yolo26n_aug_full_8419_gpu.om",
                },
            )

            state_path = Path(tmpdir.name) / "ICT" / "webui_http_unified" / "state.json"
            payload = module.json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["config"]["danger_distance"], 300)
            self.assertEqual(payload["config"]["conf_thres"], 0.55)
            self.assertEqual(payload["config"]["yolo_model"], "yolo26n_aug_full_8419_gpu.om")
            self.assertEqual(payload["capture_seq"], 7)
        finally:
            module.attach_shared_buffer = original_attach
            module.Path.home = original_home
            module.cv2.imwrite = original_imwrite
            module.json.dump = original_dump
            tmpdir.cleanup()

    def test_writer_worker_includes_marker_state_in_state_payload(self):
        module = load_module()

        class _DummyShm:
            def close(self):
                return None

        def _attach(_name, shape, dtype):
            return _DummyShm(), np.zeros(shape, dtype=dtype)

        original_attach = module.attach_shared_buffer
        original_home = module.Path.home
        original_imwrite = module.cv2.imwrite
        original_dump = module.json.dump
        module.attach_shared_buffer = _attach
        module.cv2.imwrite = lambda *_args, **_kwargs: True
        tmpdir = tempfile.TemporaryDirectory()
        module.Path.home = classmethod(lambda cls: Path(tmpdir.name))
        try:
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
            stop_event = ctx.Event()
            runtime = module.create_runtime_shared_state(ctx, laser_enabled=False)
            runtime_root = Path(tmpdir.name) / "ICT"
            runtime_root.mkdir(parents=True, exist_ok=True)
            marker_path = runtime_root / "runtime" / "unified_marker_state.json"
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text('{"selected_track_id": 5, "marker_style": "circle"}', encoding="utf-8")

            writer_queue = queue.Queue(maxsize=1)
            writer_queue.put({"fps": 9.0, "yolo_ms": 7.0, "hand_ms": 1.0, "objects": 1, "object_details": []})

            def _dump(payload, handle, *args, **kwargs):
                original_dump(payload, handle, *args, **kwargs)
                stop_event.set()

            module.json.dump = _dump

            detect_ready_evt = ctx.Event()
            detect_ready_evt.set()
            module.writer_worker(
                "color",
                "annotated",
                "overlay",
                ctx.Value("i", 9),
                ctx.Value("i", 1),
                detect_ready_evt,
                runtime,
                ["fpga"],
                writer_queue,
                stop_event,
                writer_fps=10.0,
                worker_cpu=None,
                write_debug_assets=False,
                config_snapshot={
                    "danger_distance": 300,
                    "conf_thres": 0.55,
                    "yolo_model": "det.om",
                },
            )

            state_path = runtime_root / "webui_http_unified" / "state.json"
            payload = module.json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["marker_state"]["selected_track_id"], 5)
            self.assertEqual(payload["marker_state"]["marker_style"], "circle")
        finally:
            module.attach_shared_buffer = original_attach
            module.Path.home = original_home
            module.cv2.imwrite = original_imwrite
            module.json.dump = original_dump
            tmpdir.cleanup()

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

    def test_all_production_entry_scripts_use_multiprocess_monitor(self):
        script_paths = [START_SCRIPT, LASER_RUN_SCRIPT, LASER_START_SCRIPT]

        for script_path in script_paths:
            source = script_path.read_text(encoding="utf-8")
            self.assertIn("unified_monitor_mp.py", source, msg=f"{script_path} 未切换到多进程主线")

    def test_main_start_scripts_support_board_yolom_model_directory(self):
        for script_path in (START_SCRIPT, LASER_START_SCRIPT):
            source = script_path.read_text(encoding="utf-8")
            self.assertIn(BOARD_MODEL_DIR, source, msg=f"{script_path} 未包含板卡 yolom 模型目录探测")

    def test_main_start_scripts_load_control_plane_shell_env(self):
        for script_path in (START_SCRIPT, LASER_START_SCRIPT):
            source = script_path.read_text(encoding="utf-8")
            self.assertIn("control_plane.py shell-env", source, msg=f"{script_path} 未读取统一控制配置")

    def test_main_start_scripts_prefer_yolo26n_before_yolo26m(self):
        for script_path in (START_SCRIPT, LASER_START_SCRIPT):
            source = script_path.read_text(encoding="utf-8")
            n_index = source.find('*yolo26n*.om')
            m_index = source.find('*yolo26m*.om')
            self.assertNotEqual(n_index, -1, msg=f"{script_path} 未包含 yolo26n 搜索模式")
            self.assertNotEqual(m_index, -1, msg=f"{script_path} 未包含 yolo26m 搜索模式")
            self.assertLess(n_index, m_index, msg=f"{script_path} 未优先选择 yolo26n 模型")

    def test_main_start_scripts_forward_camera_serial(self):
        for script_path in (START_SCRIPT, LASER_START_SCRIPT):
            source = script_path.read_text(encoding="utf-8")
            self.assertIn("--camera-serial", source, msg=f"{script_path} 未透传相机序列号参数")


if __name__ == "__main__":
    unittest.main()
