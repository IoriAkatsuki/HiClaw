#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend310B AIPP YOLOv8 推理 + DVPP 硬编码 + WebUI 刷新
- 输入: UVC 摄像头 (默认 /dev/video4)
- 推理: OM 模型 (AIPP) -> 直接喂 uint8 BGR
- 编码: VencSession (DVPP) 输出 H.264 码流，经 UDP 发往 --out-ip:--out-port
- WebUI: 每 0.5s 写 frame.jpg/state.json 供 http.server 查看
运行前请确保: export PYTHONPATH=~/ICT/pybind_venc/build:$PYTHONPATH
"""
import argparse
import os
import sys
import time
import json
import socket
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml
import acl
try:
    from venc_wrapper import VencSession
except ImportError as e:
    print(f"Hardware encoder disabled due to import error: {e}")
    VencSession = None

class AclLiteResource:
    def __init__(self, device_id=0):
        self.device_id = device_id
        self.context = None
        self.stream = None
    def init(self):
        ret = acl.init()
        if ret != 0:
            raise RuntimeError(f"acl.init failed: {ret}")
        ret = acl.rt.set_device(self.device_id)
        if ret != 0:
            raise RuntimeError(f"set_device failed: {ret}")
        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != 0:
            raise RuntimeError(f"create_context failed: {ret}")
        ret = acl.rt.set_context(self.context)
        if ret != 0:
            raise RuntimeError(f"set_context failed: {ret}")
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            raise RuntimeError(f"create_stream failed: {ret}")
    def release(self):
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()

class AclLiteModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_id = None
        self.desc = None
        self.input_sizes = []
        self.output_sizes = []
        self.input_buffers = []
        self.output_buffers = []
        self.host_output_buffers = []
        self.input_dataset = None
        self.output_dataset = None

    def load(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load model failed: {ret}")
        self.desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"get_desc failed: {ret}")
        self._init_io_sizes()

    def _init_io_sizes(self):
        num_inputs = acl.mdl.get_num_inputs(self.desc)
        for i in range(num_inputs):
            size = acl.mdl.get_input_size_by_index(self.desc, i)
            self.input_sizes.append(size)
            dev_ptr, ret = acl.rt.malloc(size, 2)
            if ret != 0:
                raise RuntimeError("malloc input failed")
            self.input_buffers.append(dev_ptr)
        num_outputs = acl.mdl.get_num_outputs(self.desc)
        for i in range(num_outputs):
            size = acl.mdl.get_output_size_by_index(self.desc, i)
            self.output_sizes.append(size)
            dev_ptr, ret = acl.rt.malloc(size, 2)
            if ret != 0:
                raise RuntimeError("malloc output failed")
            self.output_buffers.append(dev_ptr)
            host_ptr, ret = acl.rt.malloc_host(size)
            if ret != 0:
                raise RuntimeError("malloc_host failed")
            self.host_output_buffers.append(host_ptr)
        self.input_dataset = acl.mdl.create_dataset()
        for ptr, size in zip(self.input_buffers, self.input_sizes):
            acl.mdl.add_dataset_buffer(self.input_dataset, acl.create_data_buffer(ptr, size))
        self.output_dataset = acl.mdl.create_dataset()
        for ptr, size in zip(self.output_buffers, self.output_sizes):
            acl.mdl.add_dataset_buffer(self.output_dataset, acl.create_data_buffer(ptr, size))

    def execute(self, image_bytes):
        ret = acl.rt.memcpy(self.input_buffers[0], self.input_sizes[0],
                            acl.util.numpy_to_ptr(image_bytes), image_bytes.nbytes,
                            1)
        if ret != 0:
            return None
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            return None
        results = []
        for ptr, size, host_buf in zip(self.output_buffers, self.output_sizes, self.host_output_buffers):
            ret = acl.rt.memcpy(host_buf, size, ptr, size, 2)
            out_np = acl.util.ptr_to_numpy(host_buf, (size // 4,), 11)  # float32
            results.append(out_np.copy())
        return results

    def release(self):
        for host_buf in self.host_output_buffers:
            acl.rt.free_host(host_buf)
        self.host_output_buffers = []
        if self.model_id:
            acl.mdl.unload(self.model_id)

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def postprocess(pred_flat, ratio, dwdh, nc=61, conf_thres=0.25, iou_thres=0.45):
    ch = 4 + nc
    anchors = pred_flat.size // ch
    pred = pred_flat.reshape(1, ch, anchors)
    pred = np.transpose(pred, (0, 2, 1))[0]
    box = pred[:, :4]
    score = pred[:, 4:]
    cls_ind = np.argmax(score, axis=1)
    cls_score = np.max(score, axis=1)
    mask = cls_score >= conf_thres
    box = box[mask]; cls_ind = cls_ind[mask]; cls_score = cls_score[mask]
    if len(box) == 0:
        return []
    box_xyxy = np.zeros_like(box)
    box_xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
    box_xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
    box_xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
    box_xyxy[:, 3] = box[:, 1] + box[:, 3] / 2
    box_xyxy -= np.array(dwdh * 2)
    box_xyxy /= ratio
    indices = cv2.dnn.NMSBoxes(box_xyxy.tolist(), cls_score.tolist(), conf_thres, iou_thres)
    dets = []
    if len(indices) > 0:
        for i in indices.flatten():
            dets.append({
                'box': box_xyxy[i].tolist(),
                'score': float(cls_score[i]),
                'cls': int(cls_ind[i])
            })
            if len(dets) >= 50:
                break
    return dets


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


def safe_write_bytes(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'wb') as f:
        f.write(data)
    os.replace(tmp, path)


def safe_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w') as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def set_camera_exposure(device):
    try:
        # 强制手动曝光模式，防止画面过暗
        # auto_exposure: 1=Manual Mode
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'auto_exposure=1'], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # exposure_time_absolute: 范围 1-5000，设置 2000 保证足够明亮
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'exposure_time_absolute=2000'], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # gain: 范围 0-100，设置 64
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'gain=64'], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # brightness: 范围 -64~64，稍微调高
        subprocess.run(['v4l2-ctl', '-d', device, '-c', 'brightness=30'], 
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        print(f"Camera {device} set to Manual Exposure (exp=2000, gain=64, bri=30)")
    except Exception as e:
        print(f"Failed to set camera exposure: {e}")

def start_ffmpeg(width, height, fps, port):
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
        "-tune", "zerolatency", "-crf", "30",
        "-pix_fmt", "yuv420p", "-profile:v", "baseline", "-level", "3.0",
        "-f", "rtsp", "-rtsp_flags", "listen", f"rtsp://0.0.0.0:{port}/live"
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--cam', default='/dev/video0')
    parser.add_argument('--data-yaml', default='')
    parser.add_argument('--port', type=int, default=8554)
    parser.add_argument('--out-ip', default='127.0.0.1')
    parser.add_argument('--out-port', type=int, default=8555)
    args = parser.parse_args()

    # Explicitly set camera controls using v4l2-ctl
    set_camera_exposure(args.cam)

    res = AclLiteResource(); res.init()
    model = AclLiteModel(args.model); model.load()

    cap = cv2.VideoCapture(args.cam)
    # Removed cv2.CAP_PROP_AUTO_EXPOSURE to avoid conflict with v4l2-ctl
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        print(f"Failed to open cam {args.cam}")
        return
    w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    names = []
    if args.data_yaml:
        try:
            with open(args.data_yaml, 'r', encoding='utf-8') as f:
                names = yaml.safe_load(f).get('names', [])
        except Exception as e:
            print('load yaml failed:', e)

    enc = None
    sock = None
    target = None
    ff = None
    try:
        # Attempt to initialize NPU Hardware Encoder (DVPP)
        # Note: If this fails with error 707, it indicates internal ACL resource error.
        # Ensuring 'acl.init' is called (via res.init) is crucial.
        if VencSession is None:
             raise RuntimeError("VencSession library not available (ImportError)")
        enc = VencSession(w_cap, h_cap, codec='H264_MAIN')
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        target = (args.out_ip, args.out_port)
        print("DVPP 编码器就绪")
    except Exception as e:
        print("DVPP 编码器初始化失败，临时禁用硬编码:", e)
        enc = None
        try:
            ff = start_ffmpeg(w_cap, h_cap, 30, args.port)
            print(f"FFmpeg RTSP stream listening on port {args.port}")
        except Exception as e_ff:
            print(f"FFmpeg 启动失败: {e_ff}")

    web_dir = Path.home() / 'ICT' / 'webui_http'
    web_dir.mkdir(parents=True, exist_ok=True)
    frame_path = web_dir / 'frame.jpg'
    state_path = web_dir / 'state.json'
    last_dump = 0
    last_ts = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_in, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640))
            img_in = np.ascontiguousarray(img_in)

            t0 = time.time()
            outputs = model.execute(img_in)
            infer_ms = (time.time() - t0) * 1000

            dets = []
            if outputs:
                dets = postprocess(outputs[0], ratio, (dw, dh), conf_thres=0.15)
                h_f, w_f = frame.shape[:2]
                for d in dets:
                    x1, y1, x2, y2 = d['box']
                    x1 = max(0, min(w_f-1, int(x1)))
                    y1 = max(0, min(h_f-1, int(y1)))
                    x2 = max(0, min(w_f-1, int(x2)))
                    y2 = max(0, min(h_f-1, int(y2)))
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    label = names[d['cls']] if isinstance(names, list) and d['cls'] < len(names) else str(d['cls'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {d['score']:.2f}", (x1, max(12, y1-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            now = time.time()
            fps = 0.0
            if last_ts is not None and now > last_ts:
                fps = 1.0 / (now - last_ts)
            last_ts = now

            if enc and sock:
                try:
                    nv12 = bgr_to_nv12(frame)
                    bs = enc.encode_nv12(nv12)
                    if bs:
                        sock.sendto(bs, target)
                except Exception as e:
                    print('encode/send error:', e)
            elif ff:
                try:
                    ff.stdin.write(frame.tobytes())
                except Exception as e:
                    print("ffmpeg pipe error:", e)
                    ff = None

            if now - last_dump > 0.1:
                ok, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                if ok:
                    safe_write_bytes(frame_path, buf.tobytes())
                state = {
                    'fps': fps,
                    'infer_ms': infer_ms,
                    'dets': dets[:50],
                    'ts': now
                }
                safe_write_json(state_path, state)
                last_dump = now
                
            # Log status for debugging
            if outputs and len(dets) > 0:
                 print(f"FPS: {fps:.1f} | Infer: {infer_ms:.1f}ms | Dets: {len(dets)} | Best: {dets[0]['score']:.2f}")
            elif outputs:
                 print(f"FPS: {fps:.1f} | Infer: {infer_ms:.1f}ms | No detections")

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if enc:
            enc.close()
        if ff:
            ff.terminate()
        model.release()
        res.release()

if __name__ == '__main__':
    main()
