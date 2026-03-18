#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI Server for Unified Detection Monitor
Serves static files (HTML, images, JSON) on port 8002
"""

from __future__ import annotations

import http.server
import shutil
import socketserver
from pathlib import Path


PORT = 8002
WEB_DIR = Path.home() / "ICT" / "webui_http_unified"
TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "webui_http_unified"


def sync_template_assets(target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "debug.html"):
        src = TEMPLATE_DIR / name
        if src.exists():
            shutil.copy2(src, target_dir / name)


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        if self.command == "GET":
            return
        super().log_message(format, *args)


def main():
    sync_template_assets(WEB_DIR)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print("✓ WebUI Server 启动成功")
        print(f"  地址: http://ict.local:{PORT}")
        print(f"  目录: {WEB_DIR}")
        print("\n按 Ctrl+C 停止服务器\n")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
