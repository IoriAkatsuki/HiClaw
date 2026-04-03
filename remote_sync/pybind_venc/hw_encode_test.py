#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单验证硬编：抓取摄像头 -> NV12 -> venc_wrapper -> 写出 test.h264。
跑法：
  export PYTHONPATH=~/ICT/pybind_venc/build:$PYTHONPATH
  python3 hw_encode_test.py --cam /dev/video4 --w 640 --h 480 --frames 200
成功后会生成 ~/ICT/pybind_venc/test.h264，可用 ffplay/vlc 打开验证码流。
"""
import cv2
import argparse
import numpy as np
from venc_wrapper import VencSession
from pathlib import Path

def bgr_to_nv12(frame: np.ndarray) -> np.ndarray:
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
    ap.add_argument(--frames, type=int, default=200)
    ap.add_argument(--out, default=test.h264)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError(f"open cam {args.cam} failed")

    enc = VencSession(args.w, args.h, codec="H264_MAIN")
    out_path = Path(__file__).resolve().parent / args.out
    with open(out_path, wb) as f:
        for i in range(args.frames):
            ret, frame = cap.read()
            if not ret:
                break
            nv12 = bgr_to_nv12(frame)
            bs = enc.encode_nv12(nv12)
            f.write(bs)
            if (i+1) % 50 == 0:
                print(f"encoded {i+1}/{args.frames}")
    print(f"done, saved {out_path}")

if __name__ == __main__:
    main()
