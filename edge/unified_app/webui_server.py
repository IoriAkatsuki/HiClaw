#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI Server for Unified Detection Monitor
Serves static files (HTML, images, JSON) on port 8002
"""

from __future__ import annotations

import http.server
import json
import os
import shlex
import shutil
import signal
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse


PORT = 8002
ICT_ROOT = Path.home() / "ICT"
WEB_DIR = ICT_ROOT / "webui_http_unified"
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "webui_http_unified"
RESTART_SKIP_ENV = "ICT_SKIP_CONTROL_RESTART"

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import control_plane as cp  # noqa: E402


_calib_proc: dict = {"pid": None, "log_path": "", "proc": None, "exit_code": None}
_calib_lock = threading.Lock()


def _tail_log(log_path: str, limit: int = 20) -> str:
    path = Path(log_path)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-limit:])


def _get_calibration_status() -> dict:
    with _calib_lock:
        proc = _calib_proc.get("proc")
        running = False
        exit_code = _calib_proc.get("exit_code")
        if proc is not None:
            polled = proc.poll()
            if polled is None:
                running = True
            else:
                exit_code = int(polled)
                _calib_proc["exit_code"] = exit_code
        log_path = str(_calib_proc.get("log_path") or "")
    return {"running": running, "log_tail": _tail_log(log_path), "exit_code": None if running else exit_code}


def _start_calibration(serial_port: str, baudrate: int, output_path: str) -> dict:
    with tempfile.NamedTemporaryFile(prefix="ict_calib_", suffix=".log", delete=False) as lf:
        proc = subprocess.Popen(
            ["python3", "edge/laser_galvo/calibrate_galvo.py",
             "--serial-port", serial_port, "--baudrate", str(baudrate), "--output", output_path],
            cwd=str(ICT_ROOT), stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        log_path = lf.name
    with _calib_lock:
        _calib_proc.update({"pid": proc.pid, "log_path": log_path, "proc": proc, "exit_code": None})
    return {"pid": proc.pid, "log_path": log_path}


def sync_template_assets(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "debug.html"):
        src = TEMPLATE_DIR / name
        dst = target_dir / name
        if not src.exists():
            continue
        if dst.exists() and src.samefile(dst):
            continue
        shutil.copy2(src, dst)


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        if self.command == "GET":
            return
        super().log_message(format, *args)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _schedule_restart(self) -> dict:
        log_dir = ICT_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"control_restart_{timestamp}.log"
        if os.environ.get(RESTART_SKIP_ENV) == "1":
            return {"scheduled": False, "skipped": True, "log": str(log_path)}

        shell_cmd = (
            f"sleep 1; "
            f"pkill -f {shlex.quote('edge/unified_app/unified_monitor_mp.py')} >/dev/null 2>&1 || true; "
            f"cd {shlex.quote(str(ICT_ROOT))}; "
            f"nohup bash ./start_unified.sh > {shlex.quote(str(log_path))} 2>&1 < /dev/null &"
        )
        subprocess.Popen(
            ["/bin/bash", "-lc", shell_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        return {"scheduled": True, "log": str(log_path)}

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/control/config":
            self._send_json(cp.build_config_payload(ICT_ROOT))
            return
        if parsed.path == "/api/control/models":
            config = cp.load_runtime_config(ICT_ROOT)
            self._send_json(
                {
                    "models": cp.list_model_candidates(ICT_ROOT),
                    "current_model": config.get("yolo_model"),
                    "config": config,
                }
            )
            return
        if parsed.path == "/api/control/marker/state":
            self._send_json(cp.load_marker_state(ICT_ROOT))
            return
        if parsed.path == "/api/control/calibration/status":
            self._send_json(_get_calibration_status())
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = self._read_json_body()
            if parsed.path == "/api/control/config/apply":
                config = cp.save_runtime_config(ICT_ROOT, payload)
                self._send_json({"status": "scheduled", "config": config, "restart": self._schedule_restart()})
                return
            if parsed.path == "/api/control/models/select":
                model_path = str(payload.get("model_path", "")).strip()
                valid_paths = {item["path"] for item in cp.list_model_candidates(ICT_ROOT)}
                if model_path not in valid_paths:
                    self._send_json({"error": "模型不在允许列表中"}, status=400)
                    return
                config = cp.save_runtime_config(ICT_ROOT, {"yolo_model": model_path})
                self._send_json({"status": "scheduled", "config": config, "restart": self._schedule_restart()})
                return
            if parsed.path == "/api/control/calibration/run":
                with _calib_lock:
                    st = _get_calibration_status()
                    if st["running"]:
                        self._send_json({"error": "自动标定正在运行中", **st}, status=409)
                        return
                    config = cp.load_runtime_config(ICT_ROOT)
                    serial_port = str(payload.get("serial_port") or config.get("laser_serial") or "").strip()
                    baudrate = int(payload.get("baudrate") or config.get("laser_baudrate") or 115200)
                    output_path = str(payload.get("output") or config.get("laser_calibration") or "").strip()
                    if not serial_port:
                        self._send_json({"error": "缺少激光串口配置"}, status=400)
                        return
                    if not output_path:
                        self._send_json({"error": "缺少标定输出路径"}, status=400)
                        return
                result = _start_calibration(serial_port, baudrate, str(Path(output_path).expanduser()))
                self._send_json({"status": "started", **result})
                return
            if parsed.path == "/api/control/calibration/stop":
                with _calib_lock:
                    st = _get_calibration_status()
                    pid = _calib_proc.get("pid")
                if not pid or not st["running"]:
                    self._send_json({"status": "not_running", **st})
                    return
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                self._send_json({"status": "stopping", "pid": int(pid)})
                return
            if parsed.path == "/api/control/calibration/solve":
                self._send_json(cp.solve_homography(payload))
                return
            if parsed.path == "/api/control/calibration/apply":
                config = cp.load_runtime_config(ICT_ROOT)
                calibration_file = str(payload.get("calibration_file") or config.get("laser_calibration"))
                result = cp.apply_calibration(ICT_ROOT, calibration_file, payload)
                self._send_json({"status": "scheduled", **result, "restart": self._schedule_restart()})
                return
            if parsed.path == "/api/control/marker/class-config":
                marker_state = cp.save_marker_state(ICT_ROOT, {"class_config": payload.get("class_config", {})})
                self._send_json({"status": "ok", "marker_state": marker_state})
                return
            if parsed.path == "/api/control/marker/select":
                marker_state = cp.save_marker_state(ICT_ROOT, payload)
                self._send_json({"status": "ok", "marker_state": marker_state})
                return
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=400)
            return
        except json.JSONDecodeError:
            self._send_json({"error": "请求体不是有效 JSON"}, status=400)
            return
        except Exception as exc:
            self._send_json({"error": repr(exc)}, status=500)
            return
        self._send_json({"error": "未知接口"}, status=404)


def main():
    sync_template_assets(WEB_DIR)

    class ReusableTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("✓ WebUI Server 启动成功")
        print(f"  地址: http://ict.local:{PORT}")
        print(f"  目录: {WEB_DIR}")
        print("\n按 Ctrl+C 停止服务器\n")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
