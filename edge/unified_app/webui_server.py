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
RESTART_COOLDOWN_SECONDS = 8.0
LIVE_STATE_STALE_SECONDS = 4.0

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import control_plane as cp  # noqa: E402


_calib_proc: dict = {
    "log_path": "",
    "proc": None,
    "exit_code": None,
    "restart_after": False,
    "restart_scheduled": False,
    "restart_info": None,
}
_calib_lock = threading.RLock()
_restart_lock = threading.Lock()
_last_restart_ts = 0.0

# ── Chat backend (lazy init) ──
_chat_lock = threading.Lock()
_chat_session = None


def build_restart_shell_command(ict_root: Path, log_path: Path) -> str:
    # 用正则字符类避免 pkill 误杀承载当前 shell_cmd 的 bash 进程自身。
    target_pattern = "[u]nified_monitor_mp.py"
    python_bin = shlex.quote(sys.executable)
    pid_parts = []
    for pid in _load_runtime_worker_pids(ict_root):
        pid_parts.append(f"kill -TERM {pid} >/dev/null 2>&1 || true; ")
    return (
        f"sleep 1; "
        f"{''.join(pid_parts)}"
        f"pkill -f {shlex.quote(target_pattern)} >/dev/null 2>&1 || true; "
        f"cd {shlex.quote(str(ict_root))}; "
        f"export ICT_PYTHON_BIN={python_bin}; "
        f"nohup bash ./start_unified.sh > {shlex.quote(str(log_path))} 2>&1 < /dev/null &"
    )


def _load_runtime_worker_pids(ict_root: Path) -> list[int]:
    state = cp.load_live_state(ict_root)
    raw = state.get("worker_pids")
    if not isinstance(raw, dict):
        return []
    pids: set[int] = set()
    for value in raw.values():
        try:
            pid = int(value)
        except (TypeError, ValueError):
            continue
        if pid > 1:
            pids.add(pid)
    return list(pids)


def build_calibration_prep_shell_command(ict_root: Path) -> str:
    camera_user_patterns = (
        "[u]nified_monitor_mp.py",
        "[u]nified_monitor.py",
        "[h]and_safety_monitor.py",
        "[h]and_safety_monitor_mediapipe.py",
    )
    parts = [f"cd {shlex.quote(str(ict_root))};"]
    worker_pids = _load_runtime_worker_pids(ict_root)
    for pid in worker_pids:
        parts.append(f"kill -TERM {pid} >/dev/null 2>&1 || true;")
    for pattern in camera_user_patterns:
        parts.append(f"pkill -f {shlex.quote(pattern)} >/dev/null 2>&1 || true;")
    parts.append("sleep 1;")
    for pid in worker_pids:
        parts.append(f"kill -0 {pid} >/dev/null 2>&1 && kill -KILL {pid} >/dev/null 2>&1 || true;")
    return " ".join(parts)


def stop_monitor_for_calibration(ict_root: Path) -> None:
    shell_cmd = build_calibration_prep_shell_command(ict_root)
    subprocess.run(
        ["/bin/bash", "-lc", shell_cmd],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )


def schedule_monitor_restart(reason: str = "manual") -> dict:
    global _last_restart_ts
    log_dir = ICT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"control_restart_{timestamp}.log"
    if os.environ.get(RESTART_SKIP_ENV) == "1":
        return {"scheduled": False, "skipped": True, "reason": reason, "log": str(log_path)}

    with _restart_lock:
        now = time.time()
        if now - _last_restart_ts < RESTART_COOLDOWN_SECONDS:
            return {
                "scheduled": False,
                "cooldown": True,
                "reason": reason,
                "log": str(log_path),
                "retry_after_s": max(0.0, RESTART_COOLDOWN_SECONDS - (now - _last_restart_ts)),
            }
        shell_cmd = build_restart_shell_command(ICT_ROOT, log_path)
        subprocess.Popen(
            ["/bin/bash", "-lc", shell_cmd],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        _last_restart_ts = now
    return {"scheduled": True, "reason": reason, "log": str(log_path)}


def _monitor_process_running() -> bool:
    result = subprocess.run(
        ["/bin/bash", "-lc", "pgrep -f '[u]nified_monitor_mp.py|[s]tart_unified.sh' >/dev/null 2>&1"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _live_state_is_stale(ict_root: Path, now: float | None = None) -> bool:
    state_path = cp.get_live_state_path(ict_root)
    if not state_path.exists():
        return True
    now = time.time() if now is None else now
    try:
        age = now - state_path.stat().st_mtime
    except OSError:
        return True
    return age > LIVE_STATE_STALE_SECONDS


def _ensure_monitor_running(reason: str) -> dict | None:
    status = _get_calibration_status()
    if status.get("running"):
        return None
    if _monitor_process_running():
        return None
    if not _live_state_is_stale(ICT_ROOT):
        return None
    return schedule_monitor_restart(reason)


def _get_chat_session():
    global _chat_session
    if _chat_session is not None:
        return _chat_session
    chat_dir = str(Path(__file__).resolve().parent.parent / "qwen3_chat")
    if chat_dir not in sys.path:
        sys.path.insert(0, chat_dir)
    from chat_service import ChatSession
    _chat_session = ChatSession(max_new_tokens=128, temperature=0.0)
    return _chat_session


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
        restart_info = _calib_proc.get("restart_info")
        if proc is not None:
            polled = proc.poll()
            if polled is None:
                running = True
            else:
                exit_code, restart_info = _finalize_calibration_exit(int(polled))
        log_path = str(_calib_proc.get("log_path") or "")
    payload = {"running": running, "log_tail": _tail_log(log_path), "exit_code": None if running else exit_code}
    if restart_info is not None:
        payload["restart"] = restart_info
    return payload


def _finalize_calibration_exit(exit_code: int) -> tuple[int, dict | None]:
    with _calib_lock:
        _calib_proc["exit_code"] = int(exit_code)
        restart_info = _calib_proc.get("restart_info")
        if _calib_proc.get("restart_after") and not _calib_proc.get("restart_scheduled"):
            restart_info = schedule_monitor_restart("post_calibration")
            _calib_proc["restart_scheduled"] = True
            _calib_proc["restart_info"] = restart_info
        _calib_proc["proc"] = None
        return int(exit_code), restart_info


def _watch_calibration_process(proc: subprocess.Popen) -> None:
    try:
        exit_code = proc.wait()
    except Exception:
        return
    _finalize_calibration_exit(int(exit_code))


def _start_calibration(serial_port: str, baudrate: int, output_path: str) -> dict:
    stop_monitor_for_calibration(ICT_ROOT)
    resolved_output = Path(output_path).expanduser()
    overwrite_existing = resolved_output.exists()
    python_bin = os.environ.get("ICT_PYTHON_BIN") or sys.executable
    with tempfile.NamedTemporaryFile(prefix="ict_calib_", suffix=".log", delete=False) as lf:
        proc = subprocess.Popen(
            [python_bin, "edge/laser_galvo/calibrate_galvo.py",
             "--serial-port", serial_port, "--baudrate", str(baudrate), "--output", str(resolved_output), "--headless"],
            cwd=str(ICT_ROOT), stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
        )
        log_path = lf.name
    with _calib_lock:
        _calib_proc.update(
            {
                "log_path": log_path,
                "proc": proc,
                "exit_code": None,
                "restart_after": True,
                "restart_scheduled": False,
                "restart_info": None,
            }
        )
    watcher = threading.Thread(target=_watch_calibration_process, args=(proc,), daemon=True)
    watcher.start()
    return {
        "pid": proc.pid,
        "log_path": log_path,
        "output_path": str(resolved_output),
        "overwrite_existing": overwrite_existing,
    }


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
        return schedule_monitor_restart("config_apply")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/control/config":
            restart = _ensure_monitor_running("missing_monitor")
            payload = cp.build_config_payload(ICT_ROOT)
            if restart is not None:
                payload["auto_restart"] = restart
            self._send_json(payload)
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
        if parsed.path == "/api/chat/history":
            with _chat_lock:
                s = _chat_session
                msgs = [m for m in s.messages if m["role"] != "system"] if s else []
            self._send_json({"messages": msgs})
            return
        if parsed.path == "/api/chat/status":
            with _chat_lock:
                s = _chat_session
            try:
                chat_dir = str(Path(__file__).resolve().parent.parent / "qwen3_chat")
                if chat_dir not in sys.path:
                    sys.path.insert(0, chat_dir)
                from chat_service import BACKENDS
                backends = {k: v["label"] for k, v in BACKENDS.items()}
            except Exception:
                backends = {}
            self._send_json({
                "ready": s is not None,
                "backend": s.backend_name if s else None,
                "backends": backends,
            })
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
                    proc = _calib_proc.get("proc")
                    pid = getattr(proc, "pid", None)
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
            # ── Chat endpoints ──
            if parsed.path == "/api/chat/send":
                msg = str(payload.get("message", "")).strip()
                if not msg:
                    self._send_json({"error": "empty message"}, status=400)
                    return
                with _chat_lock:
                    s = _get_chat_session()
                    reply, n, speed = s.generate(msg)
                self._send_json({"reply": reply, "tokens": n, "speed": round(speed, 2),
                                 "backend": s.backend_name})
                return
            if parsed.path == "/api/chat/reset":
                with _chat_lock:
                    if _chat_session:
                        _chat_session.reset()
                self._send_json({"ok": True})
                return
            if parsed.path == "/api/chat/switch":
                name = str(payload.get("backend", "")).strip()
                with _chat_lock:
                    s = _get_chat_session()
                    s.switch_backend(name)
                self._send_json({"ok": True, "backend": s.backend_name})
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
