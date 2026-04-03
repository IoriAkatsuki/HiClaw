"""
YOLO26 实时检测 + 激光振镜标注 (RealSense D435)
画面显示：检测框 + 类别 + 置信度 + FPS
终端打印：每帧检测结果（可按类别过滤）
振镜输出：对启用类别画圆标注
按 Q 退出
─────────────────────────────
检测类别:
  0: fpga          1: arduino       2: stm32
  3: stlink        4: power_module  5: esp32
─────────────────────────────
坐标映射参数:
  优先使用 galvo_calibration.yaml 中的 homography_matrix
  若未提供标定文件，则退化为线性映射 K/B
"""

from pathlib import Path

import cv2
import numpy as np
import yaml


try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    YOLO = None

try:
    import pyrealsense2 as rs
except ModuleNotFoundError:
    rs = None

try:
    import serial
except ModuleNotFoundError:
    serial = None

import time


PROJECT_ROOT = Path(__file__).resolve().parent

# ============ 可调参数 ============
MODEL_PATH = str(PROJECT_ROOT / "2026_3_12/runs/train/yolo26n_aug_full_8419_gpu/weights/best.pt")
CONF_THRES = 0.75
WIDTH = 640
HEIGHT = 480

SERIAL_PORT = "COM5"
SERIAL_BAUD = 115200

CALIBRATION_FILE = str(PROJECT_ROOT / "edge/laser_galvo/galvo_calibration.yaml")

# 无标定文件时的回退参数
GALVO_KX = -100.0
GALVO_BX = 28000.0
GALVO_KY = -100.0
GALVO_BY = 19000.0

# 画圈大小: radius = RADIUS_SCALE * (bbox_w + bbox_h) / 4
RADIUS_SCALE = 120.0

PRINT_FPGA = 1
PRINT_ARDUINO = 1
PRINT_STM32 = 1
PRINT_STLINK = 1
PRINT_POWER_MODULE = 1
PRINT_ESP32 = 1
PRINT_FLAGS = [
    PRINT_FPGA,
    PRINT_ARDUINO,
    PRINT_STM32,
    PRINT_STLINK,
    PRINT_POWER_MODULE,
    PRINT_ESP32,
]
# ==================================


def load_calibration_homography(calibration_file):
    path = Path(calibration_file)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    matrix = data.get("homography_matrix")
    if matrix is None:
        return None
    return np.array(matrix, dtype=np.float32)


def px2galvo(px, py, homography=None):
    if homography is not None:
        pixel_pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        galvo_pt = cv2.perspectiveTransform(pixel_pt, homography)
        gx = int(round(float(galvo_pt[0][0][0])))
        gy = int(round(float(galvo_pt[0][0][1])))
        return gx, gy
    return int(GALVO_KX * px + GALVO_BX), int(GALVO_KY * py + GALVO_BY)


def bbox_to_circle_task(x1, y1, x2, y2, slot, homography=None):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    gx, gy = px2galvo(cx, cy, homography=homography)

    if homography is not None:
        corners = np.array(
            [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]],
            dtype=np.float32,
        )
        mapped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        width = np.linalg.norm(mapped[1] - mapped[0])
        height = np.linalg.norm(mapped[3] - mapped[0])
        r = int(max(50.0, (width + height) / 4.0))
    else:
        r = int(RADIUS_SCALE * ((x2 - x1) + (y2 - y1)) / 4.0)

    return f"{slot}C,{gx},{gy},{r}"


def main():
    if YOLO is None:
        raise ModuleNotFoundError("未找到 ultralytics，请先安装。")
    if rs is None:
        raise ModuleNotFoundError("未找到 pyrealsense2，请先安装。")
    if serial is None:
        raise ModuleNotFoundError("未找到 pyserial，请先安装。")

    homography = None
    try:
        homography = load_calibration_homography(CALIBRATION_FILE)
        if homography is not None:
            print(f"[✓] 已加载标定文件: {CALIBRATION_FILE}")
        else:
            print("[!] 未找到有效 homography，回退到线性映射")
    except Exception as exc:
        print(f"[!] 加载标定失败，回退到线性映射: {exc}")

    model = YOLO(MODEL_PATH)
    print(f"[✓] 模型已加载: {MODEL_PATH}")
    ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=1)
    print(f"[✓] 串口已连接: {SERIAL_PORT}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    pipeline.start(config)

    prev_time = time.time()
    try:
        while True:
            frame = np.asanyarray(pipeline.wait_for_frames().get_color_frame().get_data())
            results = model(frame, conf=CONF_THRES, verbose=False)[0]

            now = time.time()
            fps = 1.0 / (now - prev_time)
            prev_time = now

            detections = results.boxes
            cmd_parts = []
            slot = 0

            if len(detections):
                for box in detections:
                    cls_id = int(box.cls)
                    if not PRINT_FLAGS[cls_id]:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    cls_name = results.names[cls_id]
                    conf = float(box.conf)

                    print(f"  {cls_name}: {conf:.2f}  [{x1},{y1},{x2},{y2}]")

                    if slot < 10:
                        cmd_parts.append(
                            bbox_to_circle_task(
                                x1, y1, x2, y2, slot=slot, homography=homography
                            )
                        )
                        slot += 1

            cmd = ";".join(cmd_parts) + ";U;" if cmd_parts else "U;"
            ser.write(cmd.encode("ascii"))

            print(f"[FPS: {fps:.1f}] 检测 {len(detections)} | 标注 {slot} | → {cmd}")

            show = results.plot()
            cv2.putText(show, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("YOLO26 Detect", show)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        ser.write(b"U;")
        ser.close()
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
