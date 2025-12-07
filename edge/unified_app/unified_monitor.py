#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Detection Monitor - 电子元器件 + 手部安全
YOLOv8 物体检测 + MediaPipe Hands + RealSense D435 深度
"""
import argparse
import time
import json
import cv2
import numpy as np
import acl
import mediapipe as mp
import yaml
from pathlib import Path

class AclLiteResource:
    """ACL 资源管理"""
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
        print(f"✓ ACL 资源初始化成功 (Device {self.device_id})")

class AclLiteModel:
    """ACL 模型推理"""
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_id = None
        self.model_desc = None
        self.input_buffers = []
        self.output_buffers = []
        self.input_sizes = []
        self.output_sizes = []
        self.context = None

    def load(self):
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load_from_file failed: {ret}")
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"get_desc failed: {ret}")
        self._init_io_sizes()
        self._prepare_io_buffers()
        print(f"✓ YOLO 模型加载成功: {self.model_path}")

    def _init_io_sizes(self):
        self.input_sizes = [
            acl.mdl.get_input_size_by_index(self.model_desc, i)
            for i in range(acl.mdl.get_num_inputs(self.model_desc))
        ]
        self.output_sizes = [
            acl.mdl.get_output_size_by_index(self.model_desc, i)
            for i in range(acl.mdl.get_num_outputs(self.model_desc))
        ]

    def _prepare_io_buffers(self):
        for size in self.input_sizes:
            buf, ret = acl.rt.malloc(size, 0)
            if ret != 0:
                raise RuntimeError(f"malloc input buffer failed: {ret}")
            self.input_buffers.append(buf)
        for size in self.output_sizes:
            buf, ret = acl.rt.malloc(size, 0)
            if ret != 0:
                raise RuntimeError(f"malloc output buffer failed: {ret}")
            self.output_buffers.append(buf)

    def execute(self, image_bytes):
        acl.rt.set_context(self.context)
        ret = acl.rt.memcpy(self.input_buffers[0], self.input_sizes[0],
                           acl.util.numpy_to_ptr(image_bytes), image_bytes.nbytes, 4)
        if ret != 0:
            return None

        input_dataset = acl.mdl.create_dataset()
        input_data = acl.create_data_buffer(self.input_buffers[0], self.input_sizes[0])
        acl.mdl.add_dataset_buffer(input_dataset, input_data)

        output_dataset = acl.mdl.create_dataset()
        for i, (buf, size) in enumerate(zip(self.output_buffers, self.output_sizes)):
            output_data = acl.create_data_buffer(buf, size)
            acl.mdl.add_dataset_buffer(output_dataset, output_data)

        ret = acl.mdl.execute(self.model_id, input_dataset, output_dataset)
        if ret != 0:
            acl.mdl.destroy_dataset(input_dataset)
            acl.mdl.destroy_dataset(output_dataset)
            return None

        outputs = []
        for i, size in enumerate(self.output_sizes):
            host_buf, ret = acl.rt.malloc_host(size)
            if ret != 0:
                break
            ret = acl.rt.memcpy(host_buf, size, self.output_buffers[i], size, 3)
            if ret != 0:
                acl.rt.free_host(host_buf)
                break
            out_np = acl.util.ptr_to_numpy(host_buf, (size // 4,), 11)
            outputs.append(out_np.copy())
            acl.rt.free_host(host_buf)

        acl.mdl.destroy_dataset(input_dataset)
        acl.mdl.destroy_dataset(output_dataset)
        return outputs

def postprocess_yolo(pred_flat, ratio, dwdh, nc=61, conf_thres=0.55, iou_thres=0.45, names=None):
    """YOLOv8 后处理"""
    # 动态计算通道数和锚点数
    ch = 4 + nc  # bbox(4) + classes(nc)
    anchors = pred_flat.size // ch
    pred = pred_flat.reshape(1, ch, anchors)
    pred = np.transpose(pred, (0, 2, 1))[0]  # (anchors, ch)

    boxes_xywh = pred[:, :4]
    cls_scores_raw = pred[:, 4:]

    # Sigmoid 激活
    cls_scores = 1.0 / (1.0 + np.exp(-cls_scores_raw))

    cls_ind = cls_scores.argmax(axis=1)
    cls_score = cls_scores[np.arange(len(cls_ind)), cls_ind]

    mask = cls_score >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    cls_score = cls_score[mask]
    cls_ind = cls_ind[mask]

    if len(boxes_xywh) == 0:
        return []

    # xywh -> xyxy
    box_xyxy = np.zeros_like(boxes_xywh)
    box_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
    box_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
    box_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
    box_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

    # 缩放回原图
    box_xyxy[:, [0, 2]] = (box_xyxy[:, [0, 2]] - dwdh[0]) / ratio
    box_xyxy[:, [1, 3]] = (box_xyxy[:, [1, 3]] - dwdh[1]) / ratio

    # NMS
    from collections import defaultdict
    class_boxes = defaultdict(list)
    for i in range(len(box_xyxy)):
        class_boxes[cls_ind[i]].append((box_xyxy[i], cls_score[i], i))

    keep_indices = []
    for cls_id, boxes_list in class_boxes.items():
        if len(boxes_list) == 0:
            continue
        boxes = np.array([b[0] for b in boxes_list])
        scores = np.array([b[1] for b in boxes_list])
        original_indices = [b[2] for b in boxes_list]

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(original_indices[i])
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        keep_indices.extend(keep)

    indices = np.array(keep_indices)
    dets = []
    if len(indices) > 0:
        for i in indices:
            cls_id = int(cls_ind[i])
            cls_name = names[cls_id] if names and isinstance(names, list) and cls_id < len(names) else f"Class-{cls_id}"
            dets.append({
                'box': box_xyxy[i].tolist(),
                'score': float(cls_score[i]),
                'cls': cls_id,
                'name': cls_name
            })

    return dets

def check_hand_near_objects(wrist_pos, depth_mm, objects, danger_distance=300):
    """检查手是否接近物体"""
    if depth_mm is None or depth_mm <= 0:
        return False, None

    wx, wy = wrist_pos
    for obj in objects:
        x1, y1, x2, y2 = obj['box']
        # 检查手腕是否在物体框附近
        if x1 - 50 <= wx <= x2 + 50 and y1 - 50 <= wy <= y2 + 50:
            if depth_mm < danger_distance:
                return True, obj

    return False, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', required=True, help='YOLO OM 模型路径')
    parser.add_argument('--data-yaml', required=True, help='数据配置 YAML')
    parser.add_argument('--danger-distance', type=int, default=300, help='危险距离 (mm)')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='YOLO 置信度阈值')
    args = parser.parse_args()

    print("=" * 60)
    print("统一检测监控 - 物体 + 手部安全")
    print("=" * 60)

    # 1. 加载类别名称
    with open(args.data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get('names', [])
    print(f"✓ 加载 {len(names)} 个类别")

    # 2. 初始化 ACL
    res = AclLiteResource()
    res.init()

    # 3. 加载 YOLO 模型
    yolo_model = AclLiteModel(args.yolo_model)
    yolo_model.load()
    yolo_model.context = res.context

    # 4. 初始化 MediaPipe Hands
    print("\n初始化 MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✓ MediaPipe Hands 已初始化")

    # 5. 初始化 RealSense D435
    print("\n初始化 RealSense D435...")
    import pyrealsense2 as rs
    ctx = rs.context()
    devices = ctx.query_devices()
    if len(devices) == 0:
        raise RuntimeError("未找到 RealSense 设备")
    serial = devices[0].get_info(rs.camera_info.serial_number)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    print(f"✓ RealSense D435 已连接 ({serial})")

    # 6. WebUI 输出
    web_dir = Path.home() / 'ICT' / 'webui_http_unified'
    web_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ WebUI 输出: {web_dir}")
    print("=" * 60)
    print("\n开始监控...\n")

    last_ts = None
    frame_count = 0

    try:
        while True:
            t0 = time.time()

            # 读取相机
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            h_orig, w_orig = frame.shape[:2]

            # YOLO 推理
            img_yolo = cv2.resize(frame, (640, 640)).astype(np.uint8)
            yolo_outputs = yolo_model.execute(img_yolo)

            ratio = min(640 / w_orig, 640 / h_orig)
            new_w, new_h = int(w_orig * ratio), int(h_orig * ratio)
            dw, dh = (640 - new_w) / 2, (640 - new_h) / 2

            objects = postprocess_yolo(
                yolo_outputs[0], ratio, (dw, dh),
                nc=len(names), conf_thres=args.conf_thres, names=names
            ) if yolo_outputs else []

            yolo_ms = (time.time() - t0) * 1000

            # MediaPipe 手部检测
            t1 = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(frame_rgb)
            hand_ms = (time.time() - t1) * 1000

            # 绘制物体检测
            for obj in objects:
                x1, y1, x2, y2 = map(int, obj['box'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{obj['name']} {obj['score']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 手部检测和安全检查
            is_danger = False
            min_depth_mm = None
            hand_count = 0
            danger_obj = None

            if hand_results.multi_hand_landmarks:
                hand_count = len(hand_results.multi_hand_landmarks)
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    # 绘制完整21点骨架
                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # 获取手腕深度
                    wrist = hand_landmarks.landmark[0]
                    wx, wy = int(wrist.x * w_orig), int(wrist.y * h_orig)
                    wx = max(0, min(w_orig - 1, wx))
                    wy = max(0, min(h_orig - 1, wy))

                    try:
                        depth_m = depth_frame.get_distance(wx, wy)
                        depth_mm = depth_m * 1000
                        if depth_mm > 0:
                            if min_depth_mm is None or depth_mm < min_depth_mm:
                                min_depth_mm = depth_mm

                            # 检查手部距离是否过近
                            if depth_mm < args.danger_distance:
                                is_danger = True
                                cv2.circle(frame, (wx, wy), 15, (0, 0, 255), -1)

                                # 检查是否接近特定物体
                                near_obj, obj = check_hand_near_objects(
                                    (wx, wy), depth_mm, objects, args.danger_distance
                                )
                                if near_obj:
                                    danger_obj = obj
                                    cv2.putText(frame,
                                        f"DANGER! Hand near {obj['name']}: {depth_mm:.0f}mm",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                                else:
                                    cv2.putText(frame,
                                        f"DANGER! Hand too close: {depth_mm:.0f}mm < {args.danger_distance}mm",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                    except:
                        pass

            # 计算 FPS
            now = time.time()
            fps = 1.0 / (now - last_ts) if last_ts else 0.0
            last_ts = now

            # 状态输出
            frame_count += 1
            if frame_count % 30 == 0:
                status = "⚠ DANGER" if is_danger else "✓ SAFE"
                print(f"{status} | FPS: {fps:.1f} | YOLO: {yolo_ms:.1f}ms | Hand: {hand_ms:.1f}ms | "
                      f"物体: {len(objects)} | 手: {hand_count}")

            # WebUI 更新
            cv2.imwrite(str(web_dir / 'frame.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            state = {
                'fps': fps,
                'yolo_ms': yolo_ms,
                'hand_ms': hand_ms,
                'objects': len(objects),
                'hands': hand_count,
                'is_danger': is_danger,
                'min_depth_mm': min_depth_mm,
                'danger_object': danger_obj['name'] if danger_obj else None,
                'ts': now
            }
            with open(web_dir / 'state.json', 'w') as f:
                json.dump(state, f)

    except KeyboardInterrupt:
        print("\n\n监控已停止")
    finally:
        hands.close()
        pipeline.stop()
        print("✓ 资源已释放")

if __name__ == '__main__':
    main()
