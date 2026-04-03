#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend 310B 原生 ACL 实时检测 + RTSP 推流
流程：USB 摄像头采集 → OM 推理 → 画框 → ffmpeg rtsp_flags=listen 推流
拉流：rtsp://<板卡IP>:8554/live
"""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import yaml

# 原生 ACL
import acl


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
        self.stream, ret = acl.rt.create_stream()
        if ret != 0:
            raise RuntimeError(f"create_stream failed: {ret}")
        print(f"[NPU] Init success device={self.device_id}")

    def release(self):
        if self.stream:
            acl.rt.destroy_stream(self.stream)
        if self.context:
            acl.rt.destroy_context(self.context)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("[NPU] Released")


class AclLiteModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_id = None
        self.desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers = []
        self.output_buffers = []
        self.input_sizes = []
        self.output_sizes = []

    def load(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load model failed: {ret}")
        self.desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"get_desc failed: {ret}")
        self._init_io()
        print(f"[NPU] Model loaded: {self.model_path}")

    def _init_io(self):
        num_inputs = acl.mdl.get_num_inputs(self.desc)
        for i in range(num_inputs):
            size = acl.mdl.get_input_size_by_index(self.desc, i)
            self.input_sizes.append(size)
            buf, ret = acl.rt.malloc(size, 2)  # 2=ACL_MEM_MALLOC_HUGE_FIRST
            if ret != 0:
                raise RuntimeError("malloc input failed")
            self.input_buffers.append(buf)

        num_outputs = acl.mdl.get_num_outputs(self.desc)
        for i in range(num_outputs):
            size = acl.mdl.get_output_size_by_index(self.desc, i)
            self.output_sizes.append(size)
            buf, ret = acl.rt.malloc(size, 2)
            if ret != 0:
                raise RuntimeError("malloc output failed")
            self.output_buffers.append(buf)

        self.input_dataset = acl.mdl.create_dataset()
        for buf, size in zip(self.input_buffers, self.input_sizes):
            data_buf = acl.create_data_buffer(buf, size)
            acl.mdl.add_dataset_buffer(self.input_dataset, data_buf)

        self.output_dataset = acl.mdl.create_dataset()
        for buf, size in zip(self.output_buffers, self.output_sizes):
            data_buf = acl.create_data_buffer(buf, size)
            acl.mdl.add_dataset_buffer(self.output_dataset, data_buf)

    def execute(self, np_input: np.ndarray) -> List[np.ndarray]:
        # Host->Device
        inp_ptr = acl.util.numpy_to_ptr(np_input)
        inp_size = self.input_sizes[0]
        ret = acl.rt.memcpy(self.input_buffers[0], inp_size, inp_ptr, inp_size, 1)  # HOST_TO_DEVICE
        if ret != 0:
            raise RuntimeError("memcpy input failed")

        # Run
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            raise RuntimeError("execute failed")

        # Device->Host
        outputs = []
        for buf, size in zip(self.output_buffers, self.output_sizes):
            host_ptr, ret = acl.rt.malloc_host(size)
            if ret != 0:
                raise RuntimeError("malloc_host failed")
            ret = acl.rt.memcpy(host_ptr, size, buf, size, 2)  # DEVICE_TO_HOST
            if ret != 0:
                raise RuntimeError("memcpy output failed")
            out_np = acl.util.ptr_to_numpy(host_ptr, (size // 4,), 11)  # float32
            outputs.append(out_np.copy())
            acl.rt.free_host(host_ptr)
        return outputs

    def release(self):
        if self.model_id:
            acl.mdl.unload(self.model_id)
        for ds in [self.input_dataset, self.output_dataset]:
            if ds:
                num = acl.mdl.get_dataset_num_buffers(ds)
                for i in range(num):
                    buf = acl.mdl.get_dataset_buffer(ds, i)
                    acl.destroy_data_buffer(buf)
                acl.mdl.destroy_dataset(ds)
        for ptr in self.input_buffers + self.output_buffers:
            acl.rt.free(ptr)


def load_class_names(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        names = d.get("names", [])
        if isinstance(names, list):
            return names
    except Exception:
        pass
    return []


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw, dh = dw / 2, dh / 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)


def nms(boxes, scores, iou_thr=0.45):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]
    return keep


def postprocess(flat_pred: np.ndarray, ratio, dwdh, conf_thres=0.25, iou_thres=0.45):
    # flat_pred length -> reshape [1, 65, N]
    total = flat_pred.shape[0]
    nc_plus4 = 65
    n = total // nc_plus4
    out = flat_pred.reshape(1, nc_plus4, n)
    out = np.transpose(out, (0, 2, 1))  # [1, N, 65]
    box = out[0, :, :4]
    score = out[0, :, 4:]
    cls_ind = np.argmax(score, axis=1)
    cls_score = np.max(score, axis=1)
    mask = cls_score >= conf_thres
    box = box[mask]
    cls_ind = cls_ind[mask]
    cls_score = cls_score[mask]
    if len(box) == 0:
        return []
    box_xyxy = np.zeros_like(box)
    box_xyxy[:, 0] = box[:, 0] - box[:, 2] / 2
    box_xyxy[:, 1] = box[:, 1] - box[:, 3] / 2
    box_xyxy[:, 2] = box[:, 0] + box[:, 2] / 2
    box_xyxy[:, 3] = box[:, 1] + box[:, 3] / 2
    box_xyxy -= np.array(dwdh * 2)
    box_xyxy /= ratio
    keep = nms(box_xyxy, cls_score, iou_thres)
    dets = []
    for i in keep:
        dets.append({"box": box_xyxy[i].tolist(), "score": float(cls_score[i]), "cls": int(cls_ind[i])})
    return dets


def start_ffmpeg(width, height, fps, port):
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset", "ultrafast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-profile:v", "baseline",
        "-level", "3.0",
        "-f",
        "rtsp",
        "-rtsp_transport",
        "tcp",
        f"rtsp://127.0.0.1:{port}/live",
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description="Ascend 原生 ACL YOLOv8 RTSP")
    parser.add_argument("--model", required=True, help="OM 模型路径")
    parser.add_argument("--data-yaml", required=True, help="包含 names 的 data.yaml")
    parser.add_argument("--cam", default="/dev/video4")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--skip", type=int, default=0, help="推理跳采间隔，0 表示每帧推理")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--enable-http", action="store_true", help="是否同时输出 WebUI (静态文件刷新)")
    parser.add_argument("--http-port", type=int, default=8000)
    parser.add_argument("--web-dir", type=Path, default=Path("/home/HwHiAiUser/ICT/webui_http"))
    args = parser.parse_args()

    names = load_class_names(Path(args.data_yaml))

    res = AclLiteResource()
    res.init()
    model = AclLiteModel(args.model)
    model.load()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {args.cam}")

    ff = start_ffmpeg(args.width, args.height, args.fps, args.port)
    if ff.stdin is None:
        raise RuntimeError("启动 ffmpeg 失败")

    http_proc: Optional[subprocess.Popen] = None
    frame_path = None
    state_path = None
    if args.enable_http:
        args.web_dir.mkdir(parents=True, exist_ok=True)
        frame_path = args.web_dir / "frame.jpg"
        state_path = args.web_dir / "state.json"
        # 简易 index.html
        index_html = args.web_dir / "index.html"
        index_html.write_text(
            """
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>YOLO WebUI</title></head>
<body>
  <h3>YOLO 实时检测</h3>
  <div><img id="img" src="frame.jpg" style="max-width:90vw;"></div>
  <pre id="state">loading...</pre>
  <script>
    async function tick() {
      document.getElementById('img').src = 'frame.jpg?t=' + Date.now();
      try {
        const res = await fetch('state.json?t=' + Date.now());
        const data = await res.json();
        document.getElementById('state').innerText = JSON.stringify(data, null, 2);
      } catch (e) { console.error(e); }
      setTimeout(tick, 800);
    }
    tick();
  </script>
</body></html>
""",
            encoding="utf-8",
        )
        http_cmd = [
            "python3",
            "-m",
            "http.server",
            str(args.http_port),
            "--bind",
            "0.0.0.0",
        ]
        http_proc = subprocess.Popen(http_cmd, cwd=args.web_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[HTTP] WebUI: http://<板卡IP>:{args.http_port}/")

    def handle_sig(sig, frame):
        ff.stdin.close()
        ff.terminate()
        cap.release()
        model.release()
        res.release()
        exit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    last = time.time()
    fps_calc = 0.0
    frame_id = 0
    last_dets = []
    last_infer_ms = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue

        lb, ratio, (dw, dh) = letterbox(frame, (640, 640))
        inp = lb.transpose(2, 0, 1)[::-1]  # BGR->RGB, CHW
        inp = np.ascontiguousarray(inp, dtype=np.float32) / 255.0
        inp = np.expand_dims(inp, 0)

        do_infer = (frame_id % (args.skip + 1) == 0)
        if do_infer:
            t0 = time.time()
            outputs = model.execute(inp)
            infer_ms = (time.time() - t0) * 1000.0
            dets = postprocess(outputs[0], ratio, (dw, dh), args.conf, args.iou)
            last_dets = dets
            last_infer_ms = infer_ms
        else:
            dets = last_dets
            infer_ms = last_infer_ms
        now = time.time()
        fps_calc = 0.9 * fps_calc + 0.1 * (1.0 / (now - last)) if now != last else fps_calc
        last = now

        for d in dets:
            x1, y1, x2, y2 = map(int, d["box"])
            label = names[d["cls"]] if names and d["cls"] < len(names) else f"cls{d['cls']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {d['score']:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(frame, f"FPS:{fps_calc:.1f} infer:{infer_ms:.1f}ms det:{len(dets)}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        if args.enable_http and frame_path and state_path:
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                tmp = frame_path.with_suffix(".tmp")
                tmp.write_bytes(buf.tobytes())
                tmp.replace(frame_path)
            state = {"fps": fps_calc, "infer_ms": infer_ms, "dets": dets, "ts": time.time()}
            tmpj = state_path.with_suffix(".tmp")
            tmpj.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            tmpj.replace(state_path)

        try:
            ff.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("ffmpeg 管道关闭")
            break

        frame_id += 1

    ff.stdin.close()
    ff.terminate()
    if http_proc:
        http_proc.terminate()
    cap.release()
    model.release()
    res.release()


if __name__ == "__main__":
    main()
