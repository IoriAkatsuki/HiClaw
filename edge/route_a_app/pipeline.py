#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USB 摄像头实时检测 + WebUI 入口（路线 A）。
依赖：mindspore-lite（Ascend）、opencv-python、fastapi、uvicorn、pyyaml、numpy。

启动示例：
python edge/route_a_app/pipeline.py \
  --mindir runs/detect/train_electro61/weights/yolov8_electro61.mindir \
  --data-yaml "ElectroCom61 A Multiclass Dataset for Detection of Electronic Components/ElectroCom-61_v2/data.yaml" \
  --cam /dev/video0 --host 0.0.0.0 --port 8000
"""
from pathlib import Path
import argparse
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn

from edge.common.ms_infer.yolov8_ms import Yolov8MsInfer


def draw_boxes(frame: np.ndarray, dets: List[dict], infer_ms: float, fps: float) -> np.ndarray:
    """在图像上叠加检测框与性能信息。"""
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f'{d.get("label", "cls")}/{d["score"]:.2f}'
        cv2.putText(frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    info = f"FPS: {fps:.1f} | infer: {infer_ms:.1f} ms | dets: {len(dets)}"
    cv2.putText(frame, info, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return frame


class DetectWorker:
    """摄像头采集 + 推理循环，线程安全提供最新结果。"""

    def __init__(
        self,
        infer_engine: Yolov8MsInfer,
        cam: str,
        skip: int = 0,
        max_width: int = 1280,
        max_height: int = 720,
    ):
        self.infer_engine = infer_engine
        self.cam = cam
        self.skip = max(skip, 0)
        self.max_width = max_width
        self.max_height = max_height

        self._lock = threading.Lock()
        self._latest_jpeg: Optional[bytes] = None
        self._latest_state: Dict = {"dets": [], "infer_ms": 0.0, "fps": 0.0, "ts": time.time()}
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _loop(self):
        cap = cv2.VideoCapture(self.cam)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开摄像头 {self.cam}")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.max_height)

        frame_id = 0
        last_time = time.time()
        fps = 0.0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            if self.skip > 0 and (frame_id % (self.skip + 1)) != 0:
                dets = self._latest_state.get("dets", [])
                infer_ms = self._latest_state.get("infer_ms", 0.0)
            else:
                dets, infer_ms = self.infer_engine.infer(frame)

            now = time.time()
            dt = now - last_time
            last_time = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps

            vis = draw_boxes(frame.copy(), dets, infer_ms, fps)
            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self._lock:
                    self._latest_jpeg = buf.tobytes()
                    self._latest_state = {
                        "dets": dets,
                        "infer_ms": infer_ms,
                        "fps": fps,
                        "ts": now,
                        "cam": self.cam,
                        "model": str(self.infer_engine.mindir_path),
                    }

            frame_id += 1

        cap.release()

    def mjpeg_stream(self):
        """生成 MJPEG 流；若无新帧则短暂休眠。"""
        while True:
            with self._lock:
                jpeg = self._latest_jpeg
            if jpeg:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.03)

    def get_state(self) -> Dict:
        with self._lock:
            return dict(self._latest_state)


def create_app(worker: DetectWorker) -> FastAPI:
    app = FastAPI(title="Electro Components Live", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def index():
        html = f"""
        <html><head><title>Yolov8 NPU</title></head>
        <body>
        <h3>电子元器件检测（MindSpore NPU）</h3>
        <div>
            <img id="stream" src="/mjpeg" style="max-width: 90vw;" />
        </div>
        <pre id="info">加载中...</pre>
        <script>
        async function refresh() {{
            try {{
                const res = await fetch('/api/state');
                const data = await res.json();
                document.getElementById('info').innerText = JSON.stringify(data, null, 2);
            }} catch (e) {{
                console.error(e);
            }}
            setTimeout(refresh, 800);
        }}
        refresh();
        </script>
        </body></html>
        """
        return HTMLResponse(content=html)

    @app.get("/mjpeg")
    def mjpeg():
        return StreamingResponse(worker.mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/state")
    def api_state():
        return JSONResponse(worker.get_state())

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 NPU 实时检测 + WebUI")
    parser.add_argument("--mindir", type=Path, required=True, help="MindIR 路径（Ascend 导出）")
    parser.add_argument("--data-yaml", type=Path, required=True, help="包含 names 的 data.yaml 路径")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--cam", type=str, default="/dev/video0", help="摄像头设备或 RTSP/URL")
    parser.add_argument("--skip", type=int, default=0, help="跳采间隔（>0 表示每 N 帧推理一次）")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.45)
    return parser.parse_args()


def main():
    args = parse_args()
    infer_engine = Yolov8MsInfer(
        mindir_path=args.mindir,
        device_id=args.device_id,
        conf_thr=args.conf,
        iou_thr=args.iou,
        data_yaml=args.data_yaml,
    )
    worker = DetectWorker(infer_engine, cam=args.cam, skip=args.skip)
    worker.start()

    app = create_app(worker)
    uvicorn.run(app, host=args.host, port=args.port, reload=False, access_log=False)


if __name__ == "__main__":
    main()
