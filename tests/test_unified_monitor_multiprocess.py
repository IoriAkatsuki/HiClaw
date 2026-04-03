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
    def test_select_mp_start_method_prefers_spawn(self):
        module = load_module()

        original_get_all = module.mp.get_all_start_methods
        original_get = module.mp.get_start_method
        module.mp.get_all_start_methods = lambda: ["fork", "spawn", "forkserver"]
        module.mp.get_start_method = lambda: "fork"
        try:
            self.assertEqual(module.select_mp_start_method(), "spawn")
        finally:
            module.mp.get_all_start_methods = original_get_all
            module.mp.get_start_method = original_get

    def test_select_mp_start_method_falls_back_to_default(self):
        module = load_module()

        original_get_all = module.mp.get_all_start_methods
        original_get = module.mp.get_start_method
        module.mp.get_all_start_methods = lambda: ["fork"]
        module.mp.get_start_method = lambda: "fork"
        try:
            self.assertEqual(module.select_mp_start_method(), "fork")
        finally:
            module.mp.get_all_start_methods = original_get_all
            module.mp.get_start_method = original_get

    def test_install_stop_signal_handlers_sets_stop_event(self):
        module = load_module()
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
        stop_event = ctx.Event()

        original_getsignal = module.signal.getsignal
        original_signal = module.signal.signal
        registered = {}

        module.signal.getsignal = lambda sig: f"prev-{sig}"
        module.signal.signal = lambda sig, handler: registered.setdefault(sig, handler)
        try:
            previous = module.install_stop_signal_handlers(stop_event)
        finally:
            module.signal.getsignal = original_getsignal
            module.signal.signal = original_signal

        self.assertIn(module.signal.SIGTERM, registered)
        self.assertIn(module.signal.SIGINT, registered)
        self.assertEqual(previous[module.signal.SIGTERM], f"prev-{module.signal.SIGTERM}")
        self.assertEqual(previous[module.signal.SIGINT], f"prev-{module.signal.SIGINT}")

        registered[module.signal.SIGTERM](module.signal.SIGTERM, None)
        self.assertTrue(stop_event.is_set())

    def test_restore_signal_handlers_restores_previous_values(self):
        module = load_module()
        original_signal = module.signal.signal
        restored = []
        module.signal.signal = lambda sig, handler: restored.append((sig, handler))
        previous = {
            module.signal.SIGTERM: "term-prev",
            module.signal.SIGINT: "int-prev",
        }
        try:
            module.restore_signal_handlers(previous)
        finally:
            module.signal.signal = original_signal

        self.assertEqual(
            restored,
            [
                (module.signal.SIGTERM, "term-prev"),
                (module.signal.SIGINT, "int-prev"),
            ],
        )

    def test_get_um_module_imports_once_and_caches(self):
        module = load_module()

        original_import = module.importlib.import_module
        original_cached = getattr(module, "_UM_MODULE", None)
        calls = []

        class _DummyUm:
            pass

        def _fake_import(name):
            calls.append(name)
            self.assertEqual(name, "unified_monitor")
            return _DummyUm

        module._UM_MODULE = None
        module.importlib.import_module = _fake_import
        try:
            self.assertIs(module._get_um_module(), _DummyUm)
            self.assertIs(module._get_um_module(), _DummyUm)
        finally:
            module.importlib.import_module = original_import
            module._UM_MODULE = original_cached

        self.assertEqual(calls, ["unified_monitor"])

    def test_should_drop_stale_detection_result_allows_moderate_lag(self):
        module = load_module()

        self.assertFalse(
            module.should_drop_stale_detection_result(
                seq_captured=10,
                latest_frame_seq=14,
            )
        )

    def test_should_drop_stale_detection_result_rejects_excessive_lag(self):
        module = load_module()

        self.assertTrue(
            module.should_drop_stale_detection_result(
                seq_captured=10,
                latest_frame_seq=19,
            )
        )

    def test_emit_marker_command_uses_software_sweep_for_single_target_rectangle(self):
        module = load_module()

        class _DummyGalvo:
            def __init__(self):
                self.calls = []

            def draw_box_sweep(self, box, **kwargs):
                self.calls.append(("sweep", list(box), kwargs))
                return True

            def draw_box(self, box, **kwargs):
                self.calls.append(("box", list(box), kwargs))
                return True

            def draw_circle(self, *args, **kwargs):
                self.calls.append(("circle", args, kwargs))
                return True

            def pixel_to_galvo(self, x, y, _frame_width, _frame_height):
                return int(x), int(y)

        galvo = _DummyGalvo()

        ok = module.emit_marker_command(
            galvo,
            "rectangle",
            [10.0, 20.0, 30.0, 40.0],
            frame_width=640,
            frame_height=480,
            software_sweep=True,
        )

        self.assertTrue(ok)
        self.assertEqual(len(galvo.calls), 1)
        self.assertEqual(galvo.calls[0][0], "sweep")

    def test_emit_marker_command_keeps_multi_target_rectangle_on_lightweight_box_command(self):
        module = load_module()

        class _DummyGalvo:
            def __init__(self):
                self.calls = []

            def draw_box_sweep(self, box, **kwargs):
                self.calls.append(("sweep", list(box), kwargs))
                return True

            def draw_box(self, box, **kwargs):
                self.calls.append(("box", list(box), kwargs))
                return True

            def draw_circle(self, *args, **kwargs):
                self.calls.append(("circle", args, kwargs))
                return True

            def pixel_to_galvo(self, x, y, _frame_width, _frame_height):
                return int(x), int(y)

        galvo = _DummyGalvo()

        ok = module.emit_marker_command(
            galvo,
            "rectangle",
            [10.0, 20.0, 30.0, 40.0],
            frame_width=640,
            frame_height=480,
            software_sweep=False,
        )

        self.assertTrue(ok)
        self.assertEqual(len(galvo.calls), 1)
        self.assertEqual(galvo.calls[0][0], "box")
        self.assertNotIn("task_index", galvo.calls[0][2])



    def test_capture_worker_resolves_camera_serial_without_importing_unified_monitor(self):
        module = load_module()

        class _DummyShm:
            def close(self):
                return None

        class _DummyColorSensor:
            def set_option(self, *_args, **_kwargs):
                return None

            def get_option_range(self, _option):
                return type("_Range", (), {"min": 1, "max": 50000})()

        class _DummyDepthSensor:
            def get_depth_scale(self):
                return 0.001

        class _DummyDevice:
            def __init__(self, serial):
                self._serial = serial

            def get_info(self, _info):
                return self._serial

            def first_depth_sensor(self):
                return _DummyDepthSensor()

            def first_color_sensor(self):
                return _DummyColorSensor()

        class _DummyProfile:
            def __init__(self, serial):
                self._device = _DummyDevice(serial)

            def get_device(self):
                return self._device

        class _DummyFrame:
            def __init__(self, arr):
                self._arr = arr

            def get_data(self):
                return self._arr

        class _DummyAlignedFrames:
            def __init__(self):
                self._color = _DummyFrame(np.zeros(module.COLOR_SHAPE, dtype=np.uint8))
                self._depth = _DummyFrame(np.zeros(module.DEPTH_SHAPE, dtype=np.uint16))

            def get_color_frame(self):
                return self._color

            def get_depth_frame(self):
                return self._depth

        class _DummyAlign:
            def process(self, _frames):
                return _DummyAlignedFrames()

        class _DummyPipeline:
            def __init__(self, serial):
                self._serial = serial
                self.started = False
                self.stopped = False

            def start(self, _config):
                self.started = True
                return _DummyProfile(self._serial)

            def wait_for_frames(self):
                return object()

            def stop(self):
                self.stopped = True

        class _DummyConfig:
            def enable_device(self, serial):
                self.serial = serial

            def enable_stream(self, *_args, **_kwargs):
                return None

        class _DummyRs:
            class camera_info:
                serial_number = object()

            class stream:
                color = object()
                depth = object()

            class format:
                bgr8 = object()
                z16 = object()

            class option:
                enable_auto_exposure = object()
                exposure = object()

            def __init__(self, serial):
                self._serial = serial

            def pipeline(self):
                return _DummyPipeline(self._serial)

            def config(self):
                return _DummyConfig()

            def context(self):
                return type("_Ctx", (), {"query_devices": lambda _self: [_DummyDevice(self._serial)]})()

            def align(self, _stream):
                return _DummyAlign()

        original_attach = module.attach_shared_buffer
        original_import = module._get_um_module
        original_sleep = module.time.sleep
        original_np_copyto = module.np.copyto
        saved_rs = sys.modules.get("pyrealsense2")

        module.attach_shared_buffer = lambda _name, shape, dtype: (_DummyShm(), np.zeros(shape, dtype=dtype))
        module._get_um_module = lambda: (_ for _ in ()).throw(AssertionError("capture_worker should not import unified_monitor"))
        module.time.sleep = lambda _secs: None
        sys.modules["pyrealsense2"] = _DummyRs("SERIAL-001")

        stop_after_first_copy = {"done": False}

        def _copyto(dst, src):
            np.ndarray.__setitem__(dst, slice(None), src)
            if not stop_after_first_copy["done"]:
                stop_after_first_copy["done"] = True
                stop_event.set()

        module.np.copyto = _copyto

        try:
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
            runtime = module.create_runtime_shared_state(ctx, laser_enabled=False)
            frame_ready_event = ctx.Event()
            stop_event = ctx.Event()
            frame_seq = ctx.Value("i", 0)

            module.capture_worker(
                "color",
                "depth",
                frame_seq,
                runtime,
                frame_ready_event,
                stop_event,
                worker_cpu=None,
                preferred_camera_serial="SERIAL-001",
            )

            self.assertEqual(frame_seq.value, 1)
            self.assertEqual(module.read_text_slot(runtime.camera_serial), "SERIAL-001")
            self.assertAlmostEqual(module.get_shared_value(runtime.depth_scale), 0.001)
            self.assertTrue(frame_ready_event.is_set())
        finally:
            module.attach_shared_buffer = original_attach
            module._get_um_module = original_import
            module.time.sleep = original_sleep
            module.np.copyto = original_np_copyto
            if saved_rs is None:
                sys.modules.pop("pyrealsense2", None)
            else:
                sys.modules["pyrealsense2"] = saved_rs

    def test_laser_worker_imports_um_helpers_when_laser_enabled(self):
        module = load_module()
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
        runtime = module.create_runtime_shared_state(ctx, laser_enabled=True)
        laser_queue = queue.Queue()
        stop_event = ctx.Event()
        laser_queue.put(
            {
                "laser_objects": [
                    {
                        "track_id": 7,
                        "score": 0.99,
                        "cls": 0,
                        "name": "fpga",
                        "box": [10.0, 20.0, 30.0, 40.0],
                    }
                ],
                "is_danger": False,
                "detect_ts": 1.0,
            }
        )

        class _DummyUm:
            def select_laser_targets(self, objects, **_kwargs):
                return list(objects)

            def expand_box(self, box, _scale):
                return list(box)

            def smooth_box(self, _last_box, box, _alpha):
                return list(box)

            def should_refresh_laser_box(self, *_args, **_kwargs):
                return True

        class _DummyGalvo:
            def __init__(self, *args, **kwargs):
                self.task_index = 0

            def connect(self):
                return True

            def disconnect(self):
                return None

            def pixel_to_galvo(self, x, y, _frame_width, _frame_height):
                return int(x), int(y)

            def draw_circle(self, *args, **kwargs):
                return True

            def draw_box(self, box, **kwargs):
                return True

            def begin_batch(self):
                return None

            def update_tasks(self):
                stop_event.set()

        class _Args:
            enable_laser = True
            laser_serial = "/dev/ttyUSB0"
            laser_baudrate = 115200
            laser_calibration = "/tmp/fake_galvo.yaml"
            laser_update_interval = 0.0
            laser_force_refresh_interval = 0.01
            laser_min_score = 0.7
            max_laser_targets = 1
            laser_target_classes = None
            laser_box_scale = 1.0
            laser_smoothing_alpha = 0.0
            laser_center_threshold_px = 1.0
            laser_size_threshold_ratio = 0.01

        original_get_um = module._get_um_module
        original_load_marker_state = module.cp.load_marker_state
        original_affinity = module.apply_worker_cpu_affinity
        saved_galvo_module = sys.modules.get("galvo_controller")
        calls = {"count": 0}

        def _fake_get_um():
            calls["count"] += 1
            return _DummyUm()

        module._get_um_module = _fake_get_um
        module.cp.load_marker_state = lambda _ict_root: {
            "selected_track_id": None,
            "marker_style": "rectangle",
            "class_config": {},
        }
        module.apply_worker_cpu_affinity = lambda *_args, **_kwargs: None
        sys.modules["galvo_controller"] = type(
            "_FakeGalvoModule",
            (),
            {"LaserGalvoController": _DummyGalvo},
        )

        try:
            module.laser_worker(_Args(), runtime, laser_queue, stop_event, worker_cpu=None)
        finally:
            module._get_um_module = original_get_um
            module.cp.load_marker_state = original_load_marker_state
            module.apply_worker_cpu_affinity = original_affinity
            if saved_galvo_module is None:
                sys.modules.pop("galvo_controller", None)
            else:
                sys.modules["galvo_controller"] = saved_galvo_module

        self.assertGreaterEqual(calls["count"], 1)
        self.assertTrue(module.get_shared_bool(runtime.laser_enabled))
        self.assertTrue(module.get_shared_bool(runtime.laser_marked))
        self.assertEqual(module.read_text_slot(runtime.laser_error), "")

    def test_laser_worker_batches_multiple_targets_when_no_manual_selection(self):
        module = load_module()
        ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
        runtime = module.create_runtime_shared_state(ctx, laser_enabled=True)
        laser_queue = queue.Queue()
        stop_event = ctx.Event()
        laser_queue.put(
            {
                "laser_objects": [
                    {
                        "track_id": 7,
                        "score": 0.99,
                        "cls": 0,
                        "name": "fpga",
                        "box": [10.0, 20.0, 30.0, 40.0],
                    },
                    {
                        "track_id": 8,
                        "score": 0.95,
                        "cls": 3,
                        "name": "stlink",
                        "box": [50.0, 60.0, 90.0, 120.0],
                    },
                ],
                "is_danger": False,
                "detect_ts": 1.0,
            }
        )

        class _DummyUm:
            def select_laser_targets(self, objects, **_kwargs):
                return list(objects)

            def expand_box(self, box, _scale):
                return list(box)

            def smooth_box(self, _last_box, box, _alpha):
                return list(box)

            def should_refresh_laser_box(self, *_args, **_kwargs):
                return True

        class _DummyGalvo:
            instances = []

            def __init__(self, *args, **kwargs):
                self.task_index = 0
                self.draw_calls = []
                _DummyGalvo.instances.append(self)

            def connect(self):
                return True

            def disconnect(self):
                return None

            def pixel_to_galvo(self, x, y, _frame_width, _frame_height):
                return int(x), int(y)

            def draw_circle(self, x, y, radius, task_index=None):
                self.draw_calls.append(("circle", x, y, radius, task_index))
                return True

            def draw_box_sweep(self, box, **kwargs):
                self.draw_calls.append(("sweep", list(box), kwargs))
                return True

            def draw_box(self, box, **kwargs):
                self.draw_calls.append(("box", list(box), kwargs))
                return True

            def begin_batch(self):
                return None

            def update_tasks(self):
                stop_event.set()

        class _Args:
            enable_laser = True
            laser_serial = "/dev/ttyUSB0"
            laser_baudrate = 115200
            laser_calibration = "/tmp/fake_galvo.yaml"
            laser_update_interval = 0.0
            laser_force_refresh_interval = 0.01
            laser_min_score = 0.7
            max_laser_targets = 2
            laser_target_classes = None
            laser_box_scale = 1.0
            laser_smoothing_alpha = 0.0
            laser_center_threshold_px = 1.0
            laser_size_threshold_ratio = 0.01

        original_get_um = module._get_um_module
        original_load_marker_state = module.cp.load_marker_state
        original_affinity = module.apply_worker_cpu_affinity
        saved_galvo_module = sys.modules.get("galvo_controller")

        module._get_um_module = lambda: _DummyUm()
        module.cp.load_marker_state = lambda _ict_root: {
            "selected_track_id": None,
            "marker_style": "rectangle",
            "class_config": {},
        }
        module.apply_worker_cpu_affinity = lambda *_args, **_kwargs: None
        sys.modules["galvo_controller"] = type(
            "_FakeGalvoModuleMulti",
            (),
            {"LaserGalvoController": _DummyGalvo},
        )

        try:
            module.laser_worker(_Args(), runtime, laser_queue, stop_event, worker_cpu=None)
        finally:
            module._get_um_module = original_get_um
            module.cp.load_marker_state = original_load_marker_state
            module.apply_worker_cpu_affinity = original_affinity
            if saved_galvo_module is None:
                sys.modules.pop("galvo_controller", None)
            else:
                sys.modules["galvo_controller"] = saved_galvo_module

        self.assertEqual(len(_DummyGalvo.instances), 1)
        self.assertEqual(len(_DummyGalvo.instances[0].draw_calls), 2)
        self.assertTrue(all(call[0] == "box" for call in _DummyGalvo.instances[0].draw_calls))
        self.assertTrue(all("task_index" not in call[2] for call in _DummyGalvo.instances[0].draw_calls))
        self.assertEqual(module.read_json_slot(runtime.locked_track_ids, []), [7, 8])

    def test_build_arg_parser_disables_debug_assets_by_default(self):
        module = load_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(["--yolo-model", "det.om", "--data-yaml", "cfg.yaml"])

        self.assertFalse(args.write_debug_assets)
        self.assertEqual(args.writer_fps, 25.0)
        self.assertEqual(args.camera_serial, "")

    def test_build_arg_parser_defaults_to_multi_target_laser(self):
        module = load_module()
        parser = module.build_arg_parser()

        args = parser.parse_args(["--yolo-model", "det.om", "--data-yaml", "cfg.yaml"])

        self.assertEqual(args.max_laser_targets, 10)

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

    def test_writer_worker_skips_frame_output_when_video_stream_disabled(self):
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
        original_safe_write = module._safe_write_bytes
        module.attach_shared_buffer = _attach
        written_paths = []
        module.cv2.imwrite = lambda path, *_args, **_kwargs: written_paths.append(Path(path).name) or True
        module._safe_write_bytes = lambda path, _data: written_paths.append(Path(path).name)
        tmpdir = tempfile.TemporaryDirectory()
        module.Path.home = classmethod(lambda cls: Path(tmpdir.name))
        try:
            ctx = mp.get_context("fork" if "fork" in mp.get_all_start_methods() else mp.get_start_method())
            stop_event = ctx.Event()
            runtime = module.create_runtime_shared_state(ctx, laser_enabled=False)

            writer_queue = queue.Queue(maxsize=1)
            writer_queue.put({"fps": 15.0, "yolo_ms": 6.0, "hand_ms": 0.0, "objects": 0, "object_details": []})

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
                ctx.Value("i", 3),
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
                    "enable_video_stream": False,
                },
            )

            state_path = Path(tmpdir.name) / "ICT" / "webui_http_unified" / "state.json"
            payload = module.json.loads(state_path.read_text(encoding="utf-8"))
            self.assertFalse(payload["config"]["enable_video_stream"])
            self.assertNotIn("frame.jpg", written_paths)
            self.assertFalse((Path(tmpdir.name) / "ICT" / "webui_http_unified" / "frame.jpg").exists())
        finally:
            module.attach_shared_buffer = original_attach
            module.Path.home = original_home
            module.cv2.imwrite = original_imwrite
            module.json.dump = original_dump
            module._safe_write_bytes = original_safe_write
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

    def test_main_start_scripts_prefer_yolo26m_before_yolo26n(self):
        for script_path in (START_SCRIPT, LASER_START_SCRIPT):
            source = script_path.read_text(encoding="utf-8")
            n_index = source.find('*yolo26n*.om')
            m_index = source.find('*yolo26m*.om')
            self.assertNotEqual(n_index, -1, msg=f"{script_path} 未包含 yolo26n 搜索模式")
            self.assertNotEqual(m_index, -1, msg=f"{script_path} 未包含 yolo26m 搜索模式")
            self.assertLess(m_index, n_index, msg=f"{script_path} 未优先选择 yolo26m 模型")

    def test_main_start_scripts_forward_camera_serial(self):
        for script_path in (START_SCRIPT, LASER_START_SCRIPT):
            source = script_path.read_text(encoding="utf-8")
            self.assertIn("--camera-serial", source, msg=f"{script_path} 未透传相机序列号参数")


if __name__ == "__main__":
    unittest.main()
