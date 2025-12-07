#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebUI Server for Unified Detection Monitor
Serves static files (HTML, images, JSON) on port 8002
"""
import http.server
import socketserver
import os
from pathlib import Path

PORT = 8002
WEB_DIR = Path.home() / 'ICT' / 'webui_http_unified'

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        # Suppress logging for GET requests
        if self.command == 'GET':
            return
        super().log_message(format, *args)

def main():
    WEB_DIR.mkdir(parents=True, exist_ok=True)

    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"✓ WebUI Server 启动成功")
        print(f"  地址: http://ict.local:{PORT}")
        print(f"  目录: {WEB_DIR}")
        print(f"\n按 Ctrl+C 停止服务器\n")
        httpd.serve_forever()

if __name__ == '__main__':
    main()
