#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Safety Monitor - MediaPipe Hands + RealSense D435
专门的手部检测，实时监控手部距离并发出安全警告
当手部距离 < 150mm 时触发警告
"""
import argparse
import time
import json
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

def check_hand_safety(hand_landmarks, depth_frame, img_shape, danger_distance=150):
    """
    检查手部安全：查询手腕关键点深度

    MediaPipe Hands 21 个关键点：
      0: 手腕 (WRIST)
      4: 拇指指尖
      8: 食指指尖
      12: 中指指尖
      16: 无名指指尖
      20: 小指指尖

    返回: (is_danger, min_depth_mm, wrist_pos)
    """
    h, w = img_shape[:2]

    # 获取手腕位置 (关键点 0)
    wrist = hand_landmarks.landmark[0]
    x = int(wrist.x * w)
    y = int(wrist.y * h)

    # 边界检查
    x = max(0, min(w - 1, x))
    y = max(0, min(h - 1, y))

    # 查询深度
    try:
        depth_m = depth_frame.get_distance(x, y)
        depth_mm = depth_m * 1000  # m -> mm

        if depth_mm > 0:  # 有效深度
            is_danger = depth_mm < danger_distance
            return is_danger, depth_mm, (x, y)
    except Exception as e:
        pass  # 静默错误

    return False, None, (x, y)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--danger-distance', type=int, default=150, help='危险距离 (mm)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5, help='检测置信度阈值')
    parser.add_argument('--min-tracking-confidence', type=float, default=0.5, help='跟踪置信度阈值')
    args = parser.parse_args()

    print("=" * 60)
    print("Hand Safety Monitor - MediaPipe Hands + RealSense D435")
    print("=" * 60)

    # 1. 初始化 MediaPipe Hands
    print("\n[1/3] 初始化 MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence
    )
    print("✓ MediaPipe Hands 已初始化")
    print(f"  检测置信度: {args.min_detection_confidence}")
    print(f"  跟踪置信度: {args.min_tracking_confidence}")
    print(f"  最大手数: 2")

    # 2. 初始化 RealSense D435
    print("\n[2/3] 初始化 RealSense D435...")
    try:
        import pyrealsense2 as rs

        # 查找设备
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未找到 RealSense 设备")

        serial = devices[0].get_info(rs.camera_info.serial_number)
        print(f"  设备序列号: {serial}")

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("✓ RealSense D435 已连接")
    except Exception as e:
        print(f"✗ RealSense 初始化失败: {e}")
        return

    # 3. WebUI 输出目录
    web_dir = Path.home() / 'ICT' / 'webui_http_safety'
    web_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/3] WebUI 输出: {web_dir}")
    print("=" * 60)
    print("\n开始监控... (按 Ctrl+C 退出)\n")

    last_ts = None
    frame_count = 0

    try:
        while True:
            t0 = time.time()

            # 读取对齐的 RGB + Depth
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())

            # MediaPipe 推理 (需要 RGB 格式)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            infer_ms = (time.time() - t0) * 1000

            # 手部检测和安全检查
            is_danger = False
            min_depth_mm = None
            hand_count = 0

            if results.multi_hand_landmarks:
                hand_count = len(results.multi_hand_landmarks)

                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部关键点
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

                    # 安全检查
                    danger, depth, wrist_pos = check_hand_safety(
                        hand_landmarks, depth_frame, frame.shape, args.danger_distance
                    )

                    if depth is not None:
                        # 始终显示距离信息
                        if danger:
                            is_danger = True
                            if min_depth_mm is None or depth < min_depth_mm:
                                min_depth_mm = depth

                            # 绘制警告
                            cv2.circle(frame, wrist_pos, 15, (0, 0, 255), -1)
                            cv2.putText(
                                frame, f"DANGER! {depth:.0f}mm < {args.danger_distance}mm", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3
                            )
                        else:
                            # 显示安全距离（绿色）
                            cv2.circle(frame, wrist_pos, 10, (0, 255, 0), -1)
                            cv2.putText(
                                frame, f"Safe: {depth:.0f}mm > {args.danger_distance}mm", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
                            )
                            if min_depth_mm is None:
                                min_depth_mm = depth

            # FPS 计算
            now = time.time()
            fps = 1.0 / (now - last_ts) if last_ts else 0.0
            last_ts = now

            # 状态输出
            frame_count += 1
            if frame_count % 30 == 0:
                status = "⚠ DANGER" if is_danger else "✓ SAFE"
                depth_str = f"{min_depth_mm:.0f}mm" if min_depth_mm else "N/A"
                print(f"{status} | FPS: {fps:.1f} | 推理: {infer_ms:.1f}ms | 手数: {hand_count} | 距离: {depth_str}")

            # WebUI 更新
            if frame_count % 1 == 0:
                # 保存图像
                cv2.imwrite(str(web_dir / 'frame.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

                # 保存状态
                state = {
                    'fps': fps,
                    'infer_ms': infer_ms,
                    'hands': hand_count,
                    'is_danger': is_danger,
                    'min_depth_mm': min_depth_mm,
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
