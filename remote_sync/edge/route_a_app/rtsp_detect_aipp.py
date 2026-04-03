#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ascend310B AIPP YOLOv8 推理 + 可选 DVPP 硬编码 + WebUI 刷新。
- 输入: UVC 摄像头 (默认 /dev/video4)
- 推理: OM 模型 (AIPP) -> 直接喂 uint8 BGR
- 编码: 仅在 --enable-hw-encoder 时启用 VencSession
- WebUI: 定时写 frame.jpg/state.json 供 http.server 查看
运行前如需硬编码，请确保: export PYTHONPATH=~/ICT/pybind_venc/build:$PYTHONPATH
"""
import argparse
import json
import os
import time
from pathlib import Path

import acl
import cv2
import numpy as np
import yaml

try:
    from venc_wrapper import VencSession
except ImportError as exc:
    print(f"硬编码模块不可用，自动禁用 DVPP: {exc}")
    VencSession = None

ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _acl_ret_code(ret):
    return ret[-1] if isinstance(ret, tuple) else ret


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
        if self.stream is not None:
            try:
                acl.rt.destroy_stream(self.stream)
            except Exception:
                pass
            self.stream = None
        if self.context is not None:
            try:
                acl.rt.destroy_context(self.context)
            except Exception:
                pass
            self.context = None
        try:
            acl.rt.reset_device(self.device_id)
        except Exception:
            pass
        try:
            acl.finalize()
        except Exception:
            pass


class AclLiteModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_id = None
        self.desc = None
        self.context = None
        self.input_sizes = []
        self.output_sizes = []
        self.input_buffers = []
        self.output_buffers = []
        self.input_dataset = None
        self.output_dataset = None
        self.input_data_buffers = []
        self.output_data_buffers = []
        self.host_output_buffers = []

    def load(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load model failed: {ret}")
        self.desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"get_desc failed: {ret}")
        self._init_io_sizes()
        self._prepare_io_buffers()
        print(f"model loaded, input_sizes={self.input_sizes}, output_sizes={self.output_sizes}")

    def _init_io_sizes(self):
        self.input_sizes = [
            acl.mdl.get_input_size_by_index(self.desc, idx)
            for idx in range(acl.mdl.get_num_inputs(self.desc))
        ]
        self.output_sizes = [
            acl.mdl.get_output_size_by_index(self.desc, idx)
            for idx in range(acl.mdl.get_num_outputs(self.desc))
        ]

    def _prepare_io_buffers(self):
        for size in self.input_sizes:
            dev_ptr, ret = acl.rt.malloc(size, 2)
            if ret != 0:
                raise RuntimeError(f"malloc input failed: {ret}")
            self.input_buffers.append(dev_ptr)
        for size in self.output_sizes:
            dev_ptr, ret = acl.rt.malloc(size, 2)
            if ret != 0:
                raise RuntimeError(f"malloc output failed: {ret}")
            self.output_buffers.append(dev_ptr)

        self.input_dataset = acl.mdl.create_dataset()
        for ptr, size in zip(self.input_buffers, self.input_sizes):
            data_buffer = acl.create_data_buffer(ptr, size)
            self.input_data_buffers.append(data_buffer)
            ret = _acl_ret_code(acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer))
            if ret != 0:
                raise RuntimeError(f"add_dataset_buffer(input) failed: {ret}")

        self.output_dataset = acl.mdl.create_dataset()
        for ptr, size in zip(self.output_buffers, self.output_sizes):
            data_buffer = acl.create_data_buffer(ptr, size)
            self.output_data_buffers.append(data_buffer)
            ret = _acl_ret_code(acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer))
            if ret != 0:
                raise RuntimeError(f"add_dataset_buffer(output) failed: {ret}")

        for size in self.output_sizes:
            host_buf, ret = acl.rt.malloc_host(size)
            if ret != 0:
                raise RuntimeError(f"malloc_host failed: {ret}")
            self.host_output_buffers.append(host_buf)

    def execute(self, image_bytes):
        if self.context is not None:
            acl.rt.set_context(self.context)
        ret = acl.rt.memcpy(
            self.input_buffers[0],
            self.input_sizes[0],
            acl.util.numpy_to_ptr(image_bytes),
            image_bytes.nbytes,
            ACL_MEMCPY_HOST_TO_DEVICE,
        )
        if ret != 0:
            print(f"memcpy failed: {ret}, input_size={self.input_sizes[0]}, nbytes={image_bytes.nbytes}")
            return None

        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            print(f"execute failed: {ret}")
            return None

        results = []
        for host_buf, ptr, size in zip(self.host_output_buffers, self.output_buffers, self.output_sizes):
            ret = acl.rt.memcpy(host_buf, size, ptr, size, ACL_MEMCPY_DEVICE_TO_HOST)
            if ret != 0:
                print(f"d2h memcpy failed: {ret}")
                return None
            out_np = acl.util.ptr_to_numpy(host_buf, (size // 4,), 11)
            results.append(out_np.copy())
        return results

    def release(self):
        for data_buffer in self.input_data_buffers:
            try:
                acl.destroy_data_buffer(data_buffer)
            except Exception:
                pass
        self.input_data_buffers.clear()

        for data_buffer in self.output_data_buffers:
            try:
                acl.destroy_data_buffer(data_buffer)
            except Exception:
                pass
        self.output_data_buffers.clear()

        if self.input_dataset is not None:
            try:
                acl.mdl.destroy_dataset(self.input_dataset)
            except Exception:
                pass
            self.input_dataset = None

        if self.output_dataset is not None:
            try:
                acl.mdl.destroy_dataset(self.output_dataset)
            except Exception:
                pass
            self.output_dataset = None

        for host_buf in self.host_output_buffers:
            try:
                acl.rt.free_host(host_buf)
            except Exception:
                pass
        self.host_output_buffers.clear()

        for ptr in self.input_buffers:
            try:
                acl.rt.free(ptr)
            except Exception:
                pass
        self.input_buffers.clear()

        for ptr in self.output_buffers:
            try:
                acl.rt.free(ptr)
            except Exception:
                pass
        self.output_buffers.clear()

        if self.model_id is not None:
            try:
                acl.mdl.unload(self.model_id)
            except Exception:
                pass
            self.model_id = None

        if self.desc is not None:
            try:
                acl.mdl.destroy_desc(self.desc)
            except Exception:
                pass
            self.desc = None


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (left, top)


def postprocess(
    pred_flat,
    ratio,
    dwdh,
    nc=61,
    conf_thres=0.15,
    iou_thres=0.45,
    names=None,
    collect_debug=False,
    apply_sigmoid=False,
):
    ch = 4 + nc
    anchors = pred_flat.size // ch
    pred = pred_flat.reshape(1, ch, anchors)
    pred = np.transpose(pred, (0, 2, 1))[0]
    box = pred[:, :4]
    score = pred[:, 4:]
    if apply_sigmoid:
        score = 1.0 / (1.0 + np.exp(-score))
    cls_ind = np.argmax(score, axis=1)
    cls_score = np.max(score, axis=1)
    mask = cls_score >= conf_thres
    debug = {"cand": int(np.count_nonzero(mask)), "kept": 0} if collect_debug else None

    box = box[mask]
    cls_ind = cls_ind[mask]
    cls_score = cls_score[mask]
    if len(box) == 0:
        return [], debug

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
        for idx in indices.flatten():
            cls_id = int(cls_ind[idx])
            cls_name = names[cls_id] if names and isinstance(names, list) and cls_id < len(names) else f"Class-{cls_id}"
            dets.append({
                "box": box_xyxy[idx].tolist(),
                "score": float(cls_score[idx]),
                "cls": cls_id,
                "name": cls_name,
            })
            if len(dets) >= 50:
                break
    if debug is not None:
        debug["kept"] = len(dets)
    return dets, debug


def bgr_to_nv12(frame):
    h, w = frame.shape[:2]
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    y = yuv[0 : h * w].reshape(h, w)
    u = yuv[h * w : h * w + (h * w) // 4].reshape(h // 2, w // 2)
    v = yuv[h * w + (h * w) // 4 :].reshape(h // 2, w // 2)
    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return np.concatenate([y, uv], axis=0)


def safe_write_bytes(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def safe_write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--cam", default="/dev/video4")
    parser.add_argument("--data-yaml", default="")
    parser.add_argument("--port", type=int, default=8554)
    parser.add_argument("--out-ip", default="127.0.0.1")
    parser.add_argument("--out-port", type=int, default=8555)
    parser.add_argument("--conf-thres", type=float, default=0.15)
    parser.add_argument("--apply-sigmoid", action="store_true")
    parser.add_argument("--webui-interval", type=float, default=0.1)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--log-interval", type=float, default=2.0)
    parser.add_argument("--enable-hw-encoder", action="store_true")
    parser.add_argument("--debug-stats", action="store_true")
    args = parser.parse_args()

    ctrl_cmd = (
        f"v4l2-ctl -d {args.cam} "
        "--set-ctrl brightness=30,contrast=60,saturation=80,"
        "gain=120,exposure_auto=1,exposure_absolute=200"
    )
    os.system(ctrl_cmd + " >/dev/null 2>&1")

    res = AclLiteResource()
    model = AclLiteModel(args.model)
    cap = None
    enc = None
    sock = None
    target = None
    try:
        res.init()
        model.load()
        model.context = res.context

        cap = cv2.VideoCapture(args.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open cam {args.cam}")
        w_cap = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_cap = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if args.enable_hw_encoder and VencSession is not None:
            try:
                import socket

                enc = VencSession(w_cap, h_cap, codec="H264_MAIN")
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                target = (args.out_ip, args.out_port)
                print("DVPP 编码器就绪")
            except Exception as exc:
                print(f"DVPP 编码器初始化失败，自动禁用: {exc}")
                enc = None
                sock = None
                target = None

        names = []
        if args.data_yaml:
            try:
                data_yaml = os.path.expanduser(args.data_yaml)
                with open(data_yaml, "r", encoding="utf-8") as f:
                    names = yaml.safe_load(f).get("names", [])
            except Exception as exc:
                print(f"load yaml failed: {exc}")

        web_dir = Path.home() / "ICT" / "webui_http"
        web_dir.mkdir(parents=True, exist_ok=True)
        frame_path = web_dir / "frame.jpg"
        state_path = web_dir / "state.json"

        last_dump = 0.0
        last_log = 0.0
        last_ts = None
        fps = 0.0
        dump_interval = max(0.02, float(args.webui_interval))
        jpeg_quality = max(30, min(95, int(args.jpeg_quality)))
        log_interval = max(0.0, float(args.log_interval))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_in, ratio, (dw, dh) = letterbox(frame, new_shape=(640, 640))
            img_in = np.ascontiguousarray(img_in)

            t0 = time.perf_counter()
            outputs = model.execute(img_in)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            dets = []
            debug = None
            if outputs:
                dets, debug = postprocess(
                    outputs[0],
                    ratio,
                    (dw, dh),
                    nc=len(names) if names else 61,
                    conf_thres=args.conf_thres,
                    names=names,
                    collect_debug=args.debug_stats,
                    apply_sigmoid=args.apply_sigmoid,
                )
                h_f, w_f = frame.shape[:2]
                for det in dets:
                    x1, y1, x2, y2 = det["box"]
                    x1 = max(0, min(w_f - 1, int(x1)))
                    y1 = max(0, min(h_f - 1, int(y1)))
                    x2 = max(0, min(w_f - 1, int(x2)))
                    y2 = max(0, min(h_f - 1, int(y2)))
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    label = det.get("name") or (
                        names[det["cls"]]
                        if isinstance(names, list) and det["cls"] < len(names)
                        else str(det["cls"])
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{label} {det['score']:.2f}",
                        (x1, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            now = time.perf_counter()
            if last_ts is not None and now > last_ts:
                inst_fps = 1.0 / (now - last_ts)
                fps = inst_fps if fps == 0.0 else 0.85 * fps + 0.15 * inst_fps
            last_ts = now

            if enc and sock and target:
                try:
                    bitstream = enc.encode_nv12(bgr_to_nv12(frame))
                    if bitstream:
                        sock.sendto(bitstream, target)
                except Exception as exc:
                    print(f"encode/send error: {exc}")

            if now - last_dump >= dump_interval:
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if ok:
                    safe_write_bytes(frame_path, buf.tobytes())
                state = {
                    "fps": fps,
                    "infer_ms": infer_ms,
                    "dets": dets[:50],
                    "max_score": max((det["score"] for det in dets), default=0.0),
                    "ts": time.time(),
                }
                if args.debug_stats:
                    state.update({
                        "cand": 0 if debug is None else debug["cand"],
                        "kept": 0 if debug is None else debug["kept"],
                        "out_shape": tuple(outputs[0].shape) if outputs else (),
                        "out_max": float(outputs[0].max()) if outputs else 0.0,
                    })
                safe_write_json(state_path, state)
                last_dump = now

            if log_interval > 0 and now - last_log >= log_interval:
                if dets:
                    print(f"FPS: {fps:.1f} | Infer: {infer_ms:.1f}ms | Dets: {len(dets)} | Best: {dets[0]['score']:.2f}")
                else:
                    print(f"FPS: {fps:.1f} | Infer: {infer_ms:.1f}ms | No detections")
                last_log = now
    except KeyboardInterrupt:
        pass
    finally:
        if cap is not None:
            cap.release()
        if enc is not None:
            try:
                enc.close()
            except Exception:
                pass
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass
        model.release()
        res.release()


if __name__ == "__main__":
    main()
