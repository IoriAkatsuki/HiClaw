#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理后硬编 + RTSP 发送（示例）：
  - 读取 /dev/video4 640x480 30fps
  - NV12 -> venc_wrapper (DVPP VENC)
  - 码流通过 UDP 发到本机 8555（你可改为推送到 mediamtx RTP）
使用前：export PYTHONPATH=~/ICT/pybind_venc/build:$PYTHONPATH
"""
import cv2
import argparse
import socket
import numpy as np
from venc_wrapper import VencSession

def bgr_to_nv12(frame):
    h, w = frame.shape[:2]
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    y = yuv[0:h*w].reshape(h, w)
    u = yuv[h*w:h*w+(h*w)//4].reshape(h//2, w//2)
    v = yuv[h*w+(h*w)//4:].reshape(h//2, w//2)
    uv = np.empty((h//2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return np.concatenate([y, uv], axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(--cam, default=/dev/video4)
    ap.add_argument(--w, type=int, default=640)
    ap.add_argument(--h, type=int, default=480)
    ap.add_argument(--fps, type=int, default=30)
    ap.add_argument(--codec, default=H264_MAIN, choices=[H264_BASELINE,H264_MAIN,H264_HIGH,H265_MAIN])
    ap.add_argument(--target_ip, default=127.0.0.1)
    ap.add_argument(--target_port, type=int, default=8555)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(f"open cam {args.cam} failed")

    enc = VencSession(args.w, args.h, codec=args.codec)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.target_ip, args.target_port)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        nv12 = bgr_to_nv12(frame)
        bs = enc.encode_nv12(nv12)
        # 简单 UDP 发送示例；实际可推送到 mediamtx RTP/RTSP 输入
        try:
            sock.sendto(bs, target)
        except Exception as e:
            print(send error:, e)
            break

if __name__ == __main__:
    main()
