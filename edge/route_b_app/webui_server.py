#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Safety Monitor WebUI Server
Port 8001
"""
import http.server
import socketserver
from pathlib import Path

PORT = 8001
WEB_DIR = Path.home() / 'ICT' / 'webui_http_safety'

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def log_message(self, format, *args):
        pass  # 静默日志

if __name__ == '__main__':
    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        print(f"✓ Hand Safety WebUI 运行在 http://0.0.0.0:{PORT}")
        print(f"  文档目录: {WEB_DIR}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")
