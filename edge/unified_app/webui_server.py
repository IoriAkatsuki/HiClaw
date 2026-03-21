#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI Server for Unified Detection Monitor + Qwen3 Chat
Serves static files on port 8002 and handles chat API via SSE.
"""

from __future__ import annotations

import http.server
import json
import shutil
import socketserver
import threading
from pathlib import Path

PORT = 8002
WEB_DIR = Path.home() / "ICT" / "webui_http_unified"
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "webui_http_unified"

_chat_lock = threading.Lock()
_chat_session = None


def _get_chat_session():
    global _chat_session
    if _chat_session is not None:
        return _chat_session
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "qwen3_chat"))
        from chat_service import ChatSession
        _chat_session = ChatSession(max_new_tokens=128, temperature=0.0)
        return _chat_session
    except Exception as e:
        raise RuntimeError(f"Chat init failed: {e}")


def sync_template_assets(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    if TEMPLATE_DIR.resolve() == target_dir.resolve():
        return
    for name in ("index.html", "debug.html", "jmuxer.min.js"):
        src = TEMPLATE_DIR / name
        if src.exists():
            shutil.copy2(src, target_dir / name)


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        if self.command == "GET" and "/api/" not in (args[0] if args else ""):
            return
        super().log_message(format, *args)

    def _json_response(self, data, code=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        if self.path == "/api/chat/send":
            self._handle_chat_send()
        elif self.path == "/api/chat/reset":
            self._handle_chat_reset()
        else:
            self._json_response({"error": "not found"}, 404)

    def do_GET(self):
        if self.path == "/api/chat/history":
            self._handle_chat_history()
        elif self.path == "/api/chat/status":
            self._handle_chat_status()
        else:
            super().do_GET()

    def _handle_chat_send(self):
        body = self._read_body()
        msg = body.get("message", "").strip()
        if not msg:
            self._json_response({"error": "empty message"}, 400)
            return
        try:
            with _chat_lock:
                session = _get_chat_session()
                reply, n_tok, speed = session.generate(msg)
            self._json_response({
                "reply": reply,
                "tokens": n_tok,
                "speed": round(speed, 2),
            })
        except Exception as e:
            self._json_response({"error": str(e)}, 500)

    def _handle_chat_reset(self):
        with _chat_lock:
            global _chat_session
            if _chat_session is not None:
                _chat_session.messages = [
                    {"role": "system", "content": _chat_session.SYSTEM_PROMPT}
                ]
        self._json_response({"ok": True})

    def _handle_chat_history(self):
        with _chat_lock:
            if _chat_session is not None:
                msgs = [m for m in _chat_session.messages if m["role"] != "system"]
            else:
                msgs = []
        self._json_response({"messages": msgs})

    def _handle_chat_status(self):
        self._json_response({"ready": _chat_session is not None})


def main():
    sync_template_assets(WEB_DIR)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("✓ WebUI Server 启动成功")
        print(f"  地址: http://ict.local:{PORT}")
        print(f"  目录: {WEB_DIR}")
        print("  Chat API: /api/chat/send, /api/chat/reset, /api/chat/history")
        print("\n按 Ctrl+C 停止服务器\n")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
