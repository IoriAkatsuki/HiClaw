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

        shell_cmd = module.build_calibration_prep_shell_command(Path("/home/HwHiAiUser/ICT"))

        self.assertIn("[u]nified_monitor_mp.py", shell_cmd)
        self.assertIn("[u]nified_monitor.py", shell_cmd)
        self.assertIn("[h]and_safety_monitor.py", shell_cmd)
        self.assertIn("[h]and_safety_monitor_mediapipe.py", shell_cmd)
        self.assertIn("sleep 1", shell_cmd)

    def test_start_calibration_stops_stream_before_spawn(self):
        module = load_module()
        call_order = []

        class _DummyProc:
            pid = 43210

            def poll(self):
                return None

        original_stop = module.stop_monitor_for_calibration
        original_popen = module.subprocess.Popen
        try:
            module.stop_monitor_for_calibration = lambda ict_root: call_order.append(("stop", Path(ict_root)))

            def _fake_popen(*args, **kwargs):
                call_order.append(("spawn", args, kwargs))
                return _DummyProc()

            module.subprocess.Popen = _fake_popen

            result = module._start_calibration("/dev/ttyUSB0", 115200, "/tmp/galvo.yaml")
        finally:
            module.stop_monitor_for_calibration = original_stop
            module.subprocess.Popen = original_popen

        self.assertEqual(call_order[0][0], "stop")
        self.assertEqual(call_order[1][0], "spawn")
        self.assertIn("--headless", call_order[1][1][0])
        self.assertEqual(result["pid"], 43210)

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


if __name__ == "__main__":
    unittest.main()
