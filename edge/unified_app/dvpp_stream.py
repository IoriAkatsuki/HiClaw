#!/usr/bin/env python3
"""DVPP hardware encoding + WebSocket broadcast for unified WebUI streaming."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import select
import shutil
import struct
import subprocess
import threading
import time
from typing import List, Optional

import cv2
import numpy as np


def bgr_to_nv12(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    flat = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420).ravel()
    y = flat[0:h * w].reshape(h, w)
    u = flat[h * w:h * w + (h * w) // 4].reshape(h // 2, w // 2)
    v = flat[h * w + (h * w) // 4:].reshape(h // 2, w // 2)
    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return np.concatenate([y, uv], axis=0)


class DvppVpcResizer:
    """DVPP VPC hardware resizer with pre-allocated device buffers."""

    def __init__(self, src_w: int, src_h: int, dst_w: int, dst_h: int):
        import acl
        self._acl = acl
        self.src_w, self.src_h = src_w, src_h
        self.dst_w, self.dst_h = dst_w, dst_h

        self._stream, _ = acl.rt.create_stream()
        self._channel_desc = acl.media.dvpp_create_channel_desc()
        acl.media.dvpp_create_channel(self._channel_desc)
        self._resize_config = acl.media.dvpp_create_resize_config()

        # NV12 buffer sizes
        self._src_nv12_sz = src_h * src_w * 3 // 2
        self._dst_nv12_sz = dst_h * dst_w * 3 // 2

        self._dev_src, _ = acl.media.dvpp_malloc(self._src_nv12_sz)
        self._dev_dst, _ = acl.media.dvpp_malloc(self._dst_nv12_sz)

    def resize(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Resize BGR frame via DVPP VPC. Returns resized BGR numpy array."""
        acl = self._acl
        nv12 = bgr_to_nv12(bgr_frame)
        nv12_c = np.ascontiguousarray(nv12)
        acl.rt.memcpy(self._dev_src, self._src_nv12_sz,
                       nv12_c.ctypes.data, self._src_nv12_sz, 1)

        in_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(in_desc, self._dev_src)
        acl.media.dvpp_set_pic_desc_size(in_desc, self._src_nv12_sz)
        acl.media.dvpp_set_pic_desc_format(in_desc, 1)  # NV12
        acl.media.dvpp_set_pic_desc_width(in_desc, self.src_w)
        acl.media.dvpp_set_pic_desc_height(in_desc, self.src_h)
        acl.media.dvpp_set_pic_desc_width_stride(in_desc, self.src_w)
        acl.media.dvpp_set_pic_desc_height_stride(in_desc, self.src_h)

        out_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(out_desc, self._dev_dst)
        acl.media.dvpp_set_pic_desc_size(out_desc, self._dst_nv12_sz)
        acl.media.dvpp_set_pic_desc_format(out_desc, 1)  # NV12
        acl.media.dvpp_set_pic_desc_width(out_desc, self.dst_w)
        acl.media.dvpp_set_pic_desc_height(out_desc, self.dst_h)
        acl.media.dvpp_set_pic_desc_width_stride(out_desc, self.dst_w)
        acl.media.dvpp_set_pic_desc_height_stride(out_desc, self.dst_h)

        acl.media.dvpp_vpc_resize_async(
            self._channel_desc, in_desc, out_desc,
            self._resize_config, self._stream,
        )
        acl.rt.synchronize_stream(self._stream)

        # Read back NV12 and convert to BGR
        dst_nv12 = np.empty(self._dst_nv12_sz, dtype=np.uint8)
        acl.rt.memcpy(dst_nv12.ctypes.data, self._dst_nv12_sz,
                       self._dev_dst, self._dst_nv12_sz, 2)

        acl.media.dvpp_destroy_pic_desc(in_desc)
        acl.media.dvpp_destroy_pic_desc(out_desc)

        # NV12 → BGR
        h, w = self.dst_h, self.dst_w
        yuv = dst_nv12.reshape(h * 3 // 2, w)
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)

    def close(self):
        acl = self._acl
        try:
            acl.media.dvpp_free(self._dev_src)
            acl.media.dvpp_free(self._dev_dst)
        except Exception:
            pass
        try:
            acl.media.dvpp_destroy_resize_config(self._resize_config)
        except Exception:
            pass
        try:
            acl.media.dvpp_destroy_channel(self._channel_desc)
            acl.media.dvpp_destroy_channel_desc(self._channel_desc)
        except Exception:
            pass
        try:
            acl.rt.destroy_stream(self._stream)
        except Exception:
            pass


class DvppJpegEncoder:
    """DVPP JPEGE hardware encoder with pre-allocated device buffers."""

    def __init__(self, width: int, height: int, quality: int = 75):
        import acl
        self._acl = acl
        self.width, self.height = width, height
        nv12_size = height * width * 3 // 2
        self._nv12_size = nv12_size

        self._stream, _ = acl.rt.create_stream()
        self._channel_desc = acl.media.dvpp_create_channel_desc()
        acl.media.dvpp_create_channel(self._channel_desc)
        self._jpege_config = acl.media.dvpp_create_jpege_config()
        acl.media.dvpp_set_jpege_config_level(self._jpege_config, quality)

        # Pre-allocate reusable device buffers (avoid per-frame malloc/free)
        self._dev_input, _ = acl.media.dvpp_malloc(nv12_size)
        self._dev_output, _ = acl.media.dvpp_malloc(nv12_size)
        # Output size buffer: numpy int32 array + stable bytes backing for bytes_to_ptr
        self._out_size_arr = np.array([nv12_size], dtype=np.int32)
        self._out_bytes = self._out_size_arr.tobytes()

    def encode(self, bgr_frame: np.ndarray) -> bytes:
        acl = self._acl
        nv12 = bgr_to_nv12(bgr_frame)
        sz = self._nv12_size

        nv12_c = np.ascontiguousarray(nv12)
        acl.rt.memcpy(self._dev_input, sz, nv12_c.ctypes.data, sz, 1)  # H2D

        # Pic desc — rebuild per frame to avoid stale-ptr issues
        input_desc = acl.media.dvpp_create_pic_desc()
        acl.media.dvpp_set_pic_desc_data(input_desc, self._dev_input)
        acl.media.dvpp_set_pic_desc_size(input_desc, sz)
        acl.media.dvpp_set_pic_desc_format(input_desc, 1)  # NV12
        acl.media.dvpp_set_pic_desc_width(input_desc, self.width)
        acl.media.dvpp_set_pic_desc_height(input_desc, self.height)
        acl.media.dvpp_set_pic_desc_width_stride(input_desc, self.width)
        acl.media.dvpp_set_pic_desc_height_stride(input_desc, self.height)

        # Refresh output-size buffer with current max capacity
        self._out_bytes = self._out_size_arr.tobytes()
        out_size_ptr = acl.util.bytes_to_ptr(self._out_bytes)

        acl.media.dvpp_jpeg_encode_async(
            self._channel_desc, input_desc, self._dev_output,
            out_size_ptr, self._jpege_config, self._stream,
        )
        acl.rt.synchronize_stream(self._stream)

        out_size = int(np.frombuffer(self._out_bytes, dtype=np.int32)[0])
        jpeg_arr = np.empty(out_size, dtype=np.uint8)
        acl.rt.memcpy(jpeg_arr.ctypes.data, out_size, self._dev_output, out_size, 2)  # D2H
        acl.media.dvpp_destroy_pic_desc(input_desc)
        return jpeg_arr.tobytes()

    def close(self):
        acl = self._acl
        try:
            acl.media.dvpp_free(self._dev_input)
            acl.media.dvpp_free(self._dev_output)
        except Exception:
            pass
        try:
            acl.media.dvpp_destroy_jpege_config(self._jpege_config)
        except Exception:
            pass
        try:
            acl.media.dvpp_destroy_channel(self._channel_desc)
            acl.media.dvpp_destroy_channel_desc(self._channel_desc)
        except Exception:
            pass
        try:
            acl.rt.destroy_stream(self._stream)
        except Exception:
            pass


class VencStreamEncoder:
    """H.264 encoder via pybind11 VencSession wrapper."""

    def __init__(self, width: int, height: int, codec: str = "H264_MAIN"):
        from venc_wrapper import VencSession
        self._session = VencSession(width, height, codec)

    def encode(self, bgr_frame: np.ndarray) -> bytes:
        nv12 = bgr_to_nv12(bgr_frame)
        return self._session.encode_nv12(nv12)

    def close(self):
        try:
            self._session.close()
        except Exception:
            pass


def build_ffmpeg_h264_command(ffmpeg_bin: str, width: int, height: int, fps: int) -> List[str]:
    gop = max(int(fps * 2), 1)
    return [
        ffmpeg_bin,
        "-loglevel", "error",
        "-fflags", "nobuffer",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(max(int(fps), 1)),
        "-i", "pipe:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-threads", "1",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-g", str(gop),
        "-x264-params", f"repeat-headers=1:scenecut=0:keyint={gop}:min-keyint={gop}",
        "-f", "h264",
        "pipe:1",
    ]


class FfmpegH264Encoder:
    """Software H.264 fallback via ffmpeg/libx264 for WebSocket streaming."""

    def __init__(self, width: int, height: int, fps: int):
        ffmpeg_bin = shutil.which("ffmpeg")
        if not ffmpeg_bin:
            raise RuntimeError("ffmpeg 不可用，无法启用软件 H.264 fallback")
        self.width = width
        self.height = height
        self.fps = max(int(fps), 1)
        self._proc = subprocess.Popen(
            build_ffmpeg_h264_command(ffmpeg_bin, width, height, self.fps),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        if self._proc.stdout is None or self._proc.stdin is None:
            raise RuntimeError("ffmpeg 管道初始化失败")
        os.set_blocking(self._proc.stdout.fileno(), False)

    def _read_available(self) -> bytes:
        if self._proc.stdout is None:
            return b""
        fd = self._proc.stdout.fileno()
        chunks = []
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            try:
                chunk = os.read(fd, 1 << 20)
            except BlockingIOError:
                break
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def encode(self, bgr_frame: np.ndarray) -> bytes:
        if self._proc.poll() is not None:
            raise RuntimeError(f"ffmpeg 已退出，exit={self._proc.returncode}")
        if self._proc.stdin is None:
            raise RuntimeError("ffmpeg stdin 不可用")

        frame = np.ascontiguousarray(bgr_frame)
        self._proc.stdin.write(frame.tobytes())
        self._proc.stdin.flush()

        deadline = time.time() + 0.2
        output = bytearray()
        while time.time() < deadline:
            chunk = self._read_available()
            if chunk:
                output.extend(chunk)
                # ffmpeg 通常会在单帧后持续吐出完整 NAL；拿到一批后再短暂 drain 一次。
                time.sleep(0.002)
                continue
            if output:
                break
            time.sleep(0.002)
        return bytes(output)

    def close(self):
        proc = self._proc
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# WebSocket broadcaster (stdlib-only, no third-party deps)
# ---------------------------------------------------------------------------

_WS_MAGIC = b"258EAFA5-E914-47DA-95CA-5AB9AA86F340"


def _ws_accept_key(key: str) -> str:
    digest = hashlib.sha1(key.encode() + _WS_MAGIC).digest()
    return base64.b64encode(digest).decode()


def _ws_encode_frame(opcode: int, payload: bytes) -> bytes:
    length = len(payload)
    header = bytes([0x80 | opcode])
    if length < 126:
        header += bytes([length])
    elif length < 65536:
        header += struct.pack("!BH", 126, length)
    else:
        header += struct.pack("!BQ", 127, length)
    return header + payload


class _WSClient:
    __slots__ = ("writer", "alive")

    def __init__(self, writer: asyncio.StreamWriter):
        self.writer = writer
        self.alive = True

    async def send(self, data: bytes):
        try:
            self.writer.write(data)
            await self.writer.drain()
        except Exception:
            self.alive = False


class WebSocketBroadcaster:
    """Lightweight WebSocket server for streaming binary + text to browsers."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 10, port: int = 8003):
        self.port = port
        self.width, self.height, self.fps = width, height, fps
        self._clients: List[_WSClient] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None
        self._codec = "mjpeg"

    def set_codec(self, codec: str):
        self._codec = codec

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        except OSError as e:
            import sys
            print(f"[ws] bind failed, broadcast disabled: {e}", file=sys.stderr)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()
            self._loop = None

    async def _serve(self):
        self._server = await asyncio.start_server(
            self._handle_client, "0.0.0.0", self.port,
            reuse_address=True,
        )
        try:
            await self._server.serve_forever()
        finally:
            self._server.close()
            await self._server.wait_closed()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            request = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=5.0)
        except Exception:
            writer.close()
            return

        headers = {}
        for line in request.decode(errors="replace").split("\r\n")[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()

        ws_key = headers.get("sec-websocket-key")
        if not ws_key:
            writer.write(b"HTTP/1.1 400 Bad Request\r\n\r\n")
            writer.close()
            return

        accept = _ws_accept_key(ws_key)
        handshake = (
            "HTTP/1.1 101 Switching Protocols\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
        )
        writer.write(handshake.encode())
        await writer.drain()

        client = _WSClient(writer)
        self._clients.append(client)

        init_msg = json.dumps({
            "type": "init",
            "codec": "avc1.42E01E" if self._codec == "h264" else "mjpeg",
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        })
        await client.send(_ws_encode_frame(0x01, init_msg.encode()))

        try:
            while client.alive:
                try:
                    data = await asyncio.wait_for(reader.read(4096), timeout=30.0)
                    if not data:
                        break
                except asyncio.TimeoutError:
                    # send ping
                    await client.send(_ws_encode_frame(0x09, b""))
                except Exception:
                    break
        finally:
            client.alive = False
            try:
                writer.close()
            except Exception:
                pass
            self._clients = [c for c in self._clients if c.alive]

    def _broadcast(self, frame: bytes):
        dead = []
        for client in self._clients:
            if not client.alive:
                dead.append(client)
                continue
            loop = self._loop
            if loop and loop.is_running():
                asyncio.run_coroutine_threadsafe(client.send(frame), loop)
        if dead:
            self._clients = [c for c in self._clients if c.alive]

    def broadcast_binary(self, data: bytes):
        self._broadcast(_ws_encode_frame(0x02, data))

    def broadcast_text(self, data: str):
        self._broadcast(_ws_encode_frame(0x01, data.encode()))

    def stop(self):
        loop = self._loop
        srv = self._server
        if loop and loop.is_running() and srv:
            loop.call_soon_threadsafe(srv.close)
        if self._thread:
            self._thread.join(timeout=5.0)
        self._clients.clear()
