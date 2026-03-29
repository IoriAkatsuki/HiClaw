import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "edge" / "unified_app" / "webui_server.py"


def load_module():
    spec = importlib.util.spec_from_file_location("webui_server_restart_test", TARGET)
    module = importlib.util.module_from_spec(spec)
    sys.modules.pop("webui_server_restart_test", None)
    spec.loader.exec_module(module)
    return module


class WebUiServerRestartTest(unittest.TestCase):
    def test_build_restart_shell_command_avoids_self_match(self):
        module = load_module()

        shell_cmd = module.build_restart_shell_command(
            Path("/home/HwHiAiUser/ICT"),
            Path("/home/HwHiAiUser/ICT/logs/control_restart_test.log"),
        )

        self.assertIn("pkill -f", shell_cmd)
        self.assertIn("[u]nified_monitor_mp.py", shell_cmd)
        self.assertIn("ICT_PYTHON_BIN=", shell_cmd)
        self.assertIn(sys.executable, shell_cmd)
        self.assertIn("nohup bash ./start_unified.sh", shell_cmd)

    def test_build_calibration_prep_shell_command_stops_camera_users(self):
        module = load_module()

        original_load_live_state = module.cp.load_live_state
        try:
            module.cp.load_live_state = lambda _root: {"worker_pids": {"camera_capture": 111, "detector": 222}}
            shell_cmd = module.build_calibration_prep_shell_command(Path("/home/HwHiAiUser/ICT"))
        finally:
            module.cp.load_live_state = original_load_live_state

        self.assertIn("[u]nified_monitor_mp.py", shell_cmd)
        self.assertIn("[u]nified_monitor.py", shell_cmd)
        self.assertIn("[h]and_safety_monitor.py", shell_cmd)
        self.assertIn("[h]and_safety_monitor_mediapipe.py", shell_cmd)
        self.assertIn("kill -TERM 111", shell_cmd)
        self.assertIn("kill -TERM 222", shell_cmd)
        self.assertIn("kill -KILL 111", shell_cmd)
        self.assertIn("kill -KILL 222", shell_cmd)
        self.assertIn("sleep 1", shell_cmd)

    def test_start_calibration_stops_stream_before_spawn(self):
        module = load_module()
        call_order = []

        class _DummyProc:
            pid = 43210

            def poll(self):
                return None

            def wait(self):
                return 0

        original_stop = module.stop_monitor_for_calibration
        original_popen = module.subprocess.Popen
        original_thread = module.threading.Thread
        try:
            module.stop_monitor_for_calibration = lambda ict_root: call_order.append(("stop", Path(ict_root)))

            def _fake_popen(*args, **kwargs):
                call_order.append(("spawn", args, kwargs))
                return _DummyProc()

            class _DummyThread:
                def __init__(self, *, target=None, args=(), daemon=None):
                    call_order.append(("thread_init", target, args, daemon))

                def start(self):
                    call_order.append(("thread_start",))

            module.subprocess.Popen = _fake_popen
            module.threading.Thread = _DummyThread

            result = module._start_calibration("/dev/ttyUSB0", 115200, "/tmp/galvo.yaml")
        finally:
            module.stop_monitor_for_calibration = original_stop
            module.subprocess.Popen = original_popen
            module.threading.Thread = original_thread

        self.assertEqual(call_order[0][0], "stop")
        self.assertEqual(call_order[1][0], "spawn")
        self.assertEqual(call_order[2][0], "thread_init")
        self.assertEqual(call_order[3][0], "thread_start")
        self.assertIn("--headless", call_order[1][1][0])
        self.assertEqual(result["pid"], 43210)

    def test_start_calibration_reports_existing_output_will_be_overwritten(self):
        module = load_module()

        class _DummyProc:
            pid = 43210

            def poll(self):
                return None

            def wait(self):
                return 0

        original_stop = module.stop_monitor_for_calibration
        original_popen = module.subprocess.Popen
        original_thread = module.threading.Thread
        try:
            module.stop_monitor_for_calibration = lambda ict_root: None
            module.subprocess.Popen = lambda *args, **kwargs: _DummyProc()
            module.threading.Thread = lambda *args, **kwargs: type("_DummyThread", (), {"start": lambda self: None})()
            with module.tempfile.TemporaryDirectory() as tmpdir:
                output_path = Path(tmpdir) / "galvo.yaml"
                output_path.write_text("homography_matrix: []\n", encoding="utf-8")

                result = module._start_calibration("/dev/ttyUSB0", 115200, str(output_path))
        finally:
            module.stop_monitor_for_calibration = original_stop
            module.subprocess.Popen = original_popen
            module.threading.Thread = original_thread

        self.assertTrue(result["overwrite_existing"])
        self.assertEqual(result["output_path"], str(output_path))

    def test_calibration_lock_is_reentrant_for_nested_status_checks(self):
        module = load_module()

        self.assertTrue(module._calib_lock.acquire(timeout=0.1))
        try:
            status = module._get_calibration_status()
        finally:
            module._calib_lock.release()

        self.assertIn("running", status)
        self.assertIn("exit_code", status)

    def test_get_calibration_status_restarts_monitor_after_completion(self):
        module = load_module()
        restart_calls = []

        class _FinishedProc:
            def poll(self):
                return 0

        original_schedule = module.schedule_monitor_restart
        original_state = dict(module._calib_proc)
        try:
            module.schedule_monitor_restart = lambda reason="": restart_calls.append(reason) or {
                "scheduled": True,
                "reason": reason,
                "log": "/tmp/restart.log",
            }
            module._calib_proc.update(
                {
                    "pid": 123,
                    "log_path": "/tmp/test.log",
                    "proc": _FinishedProc(),
                    "exit_code": None,
                    "restart_after": True,
                    "restart_scheduled": False,
                    "restart_info": None,
                }
            )

            status = module._get_calibration_status()
        finally:
            module.schedule_monitor_restart = original_schedule
            module._calib_proc.clear()
            module._calib_proc.update(original_state)

        self.assertFalse(status["running"])
        self.assertEqual(status["exit_code"], 0)
        self.assertEqual(restart_calls, ["post_calibration"])
        self.assertEqual(status["restart"]["reason"], "post_calibration")

    def test_finalize_calibration_exit_restarts_without_status_polling(self):
        module = load_module()
        restart_calls = []

        original_schedule = module.schedule_monitor_restart
        original_state = dict(module._calib_proc)
        try:
            module.schedule_monitor_restart = lambda reason="": restart_calls.append(reason) or {
                "scheduled": True,
                "reason": reason,
                "log": "/tmp/restart.log",
            }
            module._calib_proc.update(
                {
                    "log_path": "/tmp/test.log",
                    "proc": object(),
                    "exit_code": None,
                    "restart_after": True,
                    "restart_scheduled": False,
                    "restart_info": None,
                }
            )

            exit_code, restart = module._finalize_calibration_exit(0)
        finally:
            module.schedule_monitor_restart = original_schedule
            module._calib_proc.clear()
            module._calib_proc.update(original_state)

        self.assertEqual(exit_code, 0)
        self.assertEqual(restart_calls, ["post_calibration"])
        self.assertEqual(restart["reason"], "post_calibration")

    def test_calibration_stop_uses_proc_pid_without_cached_pid_field(self):
        module = load_module()
        sent = {}
        kill_calls = []

        class _RunningProc:
            pid = 56789

        original_status = module._get_calibration_status
        original_kill = module.os.kill
        original_state = dict(module._calib_proc)
        try:
            module._get_calibration_status = lambda: {"running": True, "log_tail": "", "exit_code": None}
            module.os.kill = lambda pid, sig: kill_calls.append((pid, sig))
            module._calib_proc.clear()
            module._calib_proc.update(
                {
                    "log_path": "/tmp/test.log",
                    "proc": _RunningProc(),
                    "exit_code": None,
                    "restart_after": True,
                    "restart_scheduled": False,
                    "restart_info": None,
                }
            )

            handler = module.MyHTTPRequestHandler.__new__(module.MyHTTPRequestHandler)
            handler.path = "/api/control/calibration/stop"
            handler._read_json_body = lambda: {}
            handler._send_json = lambda payload, status=200: sent.update({"payload": payload, "status": status})

            handler.do_POST()
        finally:
            module._get_calibration_status = original_status
            module.os.kill = original_kill
            module._calib_proc.clear()
            module._calib_proc.update(original_state)

        self.assertEqual(sent["status"], 200)
        self.assertEqual(sent["payload"]["status"], "stopping")
        self.assertEqual(sent["payload"]["pid"], 56789)
        self.assertEqual(kill_calls, [(56789, module.signal.SIGTERM)])


if __name__ == "__main__":
    unittest.main()
