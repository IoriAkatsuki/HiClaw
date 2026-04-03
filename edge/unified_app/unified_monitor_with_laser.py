#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一检测监控 + 激光振镜标注
在unified_monitor.py基础上增加激光打框功能
"""
import sys
sys.path.append('/home/oasis/Documents/ICT/edge/laser_galvo')

from galvo_controller import LaserGalvoController
import argparse
import time
import json
import cv2
import numpy as np
import acl
import mediapipe as mp
import yaml
from pathlib import Path

# 导入unified_monitor的类
sys.path.append('/home/oasis/Documents/ICT/edge/unified_app')
from unified_monitor import AclLiteResource, AclLiteModel, postprocess_yolo, check_hand_near_objects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-model', required=True, help='YOLO OM 模型路径')
    parser.add_argument('--data-yaml', required=True, help='数据配置 YAML')
    parser.add_argument('--danger-distance', type=int, default=300, help='危险距离 (mm)')
    parser.add_argument('--conf-thres', type=float, default=0.55, help='YOLO 置信度阈值')

    # 激光振镜参数
    parser.add_argument('--enable-laser', action='store_true', help='启用激光打框')
    parser.add_argument('--laser-serial', default='/dev/ttyUSB0', help='激光串口设备')
    parser.add_argument('--laser-baudrate', type=int, default=115200, help='激光串口波特率')
    parser.add_argument('--laser-calibration', help='激光标定文件')
    parser.add_argument('--laser-target-classes', nargs='+', help='需要激光标注的类别名称列表')
    parser.add_argument('--laser-min-score', type=float, default=0.7, help='激光标注最小置信度')

    args = parser.parse_args()

    print("=" * 60)
    print("统一检测监控 + 激光振镜标注")
    print("=" * 60)

    # 1. 加载类别名称
    with open(args.data_yaml, 'r') as f:
        data_cfg = yaml.safe_load(f)
    names = data_cfg.get('names', [])
    print(f"✓ 加载 {len(names)} 个类别")

    # 2. 初始化激光振镜（如果启用）
    laser_controller = None
    if args.enable_laser:
        print("\n初始化激光振镜...")
        if not args.laser_calibration:
            print("✗ 需要提供标定文件 (--laser-calibration)")
            return

        laser_controller = LaserGalvoController(
            serial_port=args.laser_serial,
            baudrate=args.laser_baudrate,
            calibration_file=args.laser_calibration
        )

        if not laser_controller.connect():
            print("✗ 激光振镜连接失败，继续运行但不使用激光功能")
            laser_controller = None
        else:
            print("✓ 激光振镜已就绪")

            # 确定目标类别
            if args.laser_target_classes:
                print(f"✓ 激光标注类别: {', '.join(args.laser_target_classes)}")
            else:
                print("✓ 激光标注所有检测到的物体")

    # 3. 初始化 ACL
    print("\n初始化 ACL...")
    res = AclLiteResource()
    res.init()

    # 4. 加载 YOLO 模型
    print("\n初始化 YOLO 模型...")
    yolo_model = AclLiteModel(args.yolo_model)
    yolo_model.load()
    yolo_model.context = res.context

    # 5. 初始化 MediaPipe Hands
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

    # 6. 初始化 RealSense D435
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

    # 7. WebUI 输出
    web_dir = Path.home() / 'ICT' / 'webui_http_unified'
    web_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n✓ WebUI 输出: {web_dir}")
    print("=" * 60)
    print("\n开始监控...\\n")

    last_ts = None
    frame_count = 0
    last_laser_time = 0
    laser_cooldown = 2.0  # 激光打框冷却时间（秒）

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
                    # 绘制骨架
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

            # 激光打框逻辑
            laser_marked = False
            if laser_controller and not is_danger:  # 只在安全时使用激光
                # 检查冷却时间
                current_time = time.time()
                if current_time - last_laser_time >= laser_cooldown:
                    # 筛选需要标注的物体
                    targets = []
                    for obj in objects:
                        # 置信度过滤
                        if obj['score'] < args.laser_min_score:
                            continue

                        # 类别过滤
                        if args.laser_target_classes and obj['name'] not in args.laser_target_classes:
                            continue

                        targets.append(obj)

                    # 标注目标（限制数量避免过长时间）
                    if targets:
                        max_targets = 3  # 最多标注3个
                        for obj in targets[:max_targets]:
                            try:
                                box = obj['box']
                                laser_controller.draw_box(box, steps_per_edge=15, pixel_coords=True)
                                laser_marked = True
                                time.sleep(0.05)  # 每个框之间短暂延迟
                            except Exception as e:
                                print(f"激光打框失败: {e}")
                                break

                        if laser_marked:
                            last_laser_time = current_time
                            # 在显示图像上标记已激光标注
                            for obj in targets[:max_targets]:
                                x1, y1, x2, y2 = map(int, obj['box'])
                                cv2.putText(frame, "LASER MARKED", (x1, y2 + 20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 计算 FPS
            now = time.time()
            fps = 1.0 / (now - last_ts) if last_ts else 0.0
            last_ts = now

            # 状态输出
            frame_count += 1
            if frame_count % 30 == 0:
                status = "⚠ DANGER" if is_danger else "✓ SAFE"
                laser_status = " | 激光: ON" if laser_marked else ""
                print(f"{status} | FPS: {fps:.1f} | YOLO: {yolo_ms:.1f}ms | Hand: {hand_ms:.1f}ms | "
                      f"物体: {len(objects)} | 手: {hand_count}{laser_status}")

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
                'laser_enabled': laser_controller is not None,
                'laser_marked': laser_marked,
                'ts': now
            }
            with open(web_dir / 'state.json', 'w') as f:
                json.dump(state, f)

    except KeyboardInterrupt:
        print("\n\n监控已停止")
    finally:
        hands.close()
        pipeline.stop()
        if laser_controller:
            laser_controller.disconnect()
        print("✓ 资源已释放")


if __name__ == '__main__':
    main()
