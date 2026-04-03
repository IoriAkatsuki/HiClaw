#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Safety Monitor - YOLOv8-pose + RealSense D435
实时检测手部位置，结合深度信息进行安全警告
当手部距离物体 < 15cm 时触发警告
"""
import argparse
import time
import json
import cv2
import numpy as np
import acl
from pathlib import Path

# TODO: 需要安装 pyrealsense2
# pip install pyrealsense2

class AclLiteResource:
    """ACL 资源管理（复用 route_a 的实现）"""
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
    """ACL 模型推理（复用 route_a 的实现）"""
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_id = None
        self.model_desc = None
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffers = []
        self.output_buffers = []
        self.input_sizes = []
        self.output_sizes = []
        self.input_data_buffers = []
        self.output_data_buffers = []
        self.host_output_buffers = []
        self.context = None

    def load(self):
        """加载模型"""
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != 0:
            raise RuntimeError(f"load_from_file failed: {ret}")
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != 0:
            raise RuntimeError(f"get_desc failed: {ret}")
        self._init_io_sizes()
        self._prepare_io_buffers()
        print(f"✓ 模型加载成功: {self.model_path}")
        print(f"  输入大小: {self.input_sizes}")
        print(f"  输出大小: {self.output_sizes}")

    def _init_io_sizes(self):
        """初始化输入输出大小"""
        self.input_sizes = [
            acl.mdl.get_input_size_by_index(self.model_desc, i)
            for i in range(acl.mdl.get_num_inputs(self.model_desc))
        ]
        self.output_sizes = [
            acl.mdl.get_output_size_by_index(self.model_desc, i)
            for i in range(acl.mdl.get_num_outputs(self.model_desc))
        ]

    def _prepare_io_buffers(self):
        """准备输入输出缓冲区"""
        self.input_dataset = acl.mdl.create_dataset()
        self.output_dataset = acl.mdl.create_dataset()
        self.input_data_buffers = []
        self.output_data_buffers = []
        self.host_output_buffers = []

        for size in self.input_sizes:
            buf, ret = acl.rt.malloc(size, 0)  # ACL_MEM_MALLOC_HUGE_FIRST
            if ret != 0:
                raise RuntimeError(f"malloc input buffer failed: {ret}")
            self.input_buffers.append(buf)
            data_buffer = acl.create_data_buffer(buf, size)
            acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)
            self.input_data_buffers.append(data_buffer)

        for size in self.output_sizes:
            buf, ret = acl.rt.malloc(size, 0)
            if ret != 0:
                raise RuntimeError(f"malloc output buffer failed: {ret}")
            self.output_buffers.append(buf)
            data_buffer = acl.create_data_buffer(buf, size)
            acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer)
            self.output_data_buffers.append(data_buffer)

            host_buf = None
            malloc_host = getattr(acl.rt, "malloc_host", None)
            if malloc_host is not None:
                host_buf, ret = malloc_host(size)
                if ret != 0:
                    raise RuntimeError(f"malloc_host failed: {ret}")
            self.host_output_buffers.append(host_buf)

    def execute(self, image_bytes):
        """执行推理"""
        acl.rt.set_context(self.context)

        # 输入数据拷贝
        ret = acl.rt.memcpy(self.input_buffers[0], self.input_sizes[0],
                           acl.util.numpy_to_ptr(image_bytes), image_bytes.nbytes,
                           4)  # ACL_MEMCPY_HOST_TO_DEVICE
        if ret != 0:
            print(f"memcpy failed: {ret}")
            return None

        # 执行推理
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        if ret != 0:
            print(f"execute failed: {ret}")
            return None

        # 获取输出
        outputs = []
        for i, size in enumerate(self.output_sizes):
            host_buf = self.host_output_buffers[i] if i < len(self.host_output_buffers) else None
            ephemeral = False
            if host_buf is None:
                host_buf, ret = acl.rt.malloc_host(size)
                if ret != 0:
                    break
                ephemeral = True
            ret = acl.rt.memcpy(host_buf, size, self.output_buffers[i], size, 3)  # DEVICE_TO_HOST
            if ret != 0:
                if ephemeral:
                    acl.rt.free_host(host_buf)
                break
            out_np = acl.util.ptr_to_numpy(host_buf, (size // 4,), 11)  # float32
            outputs.append(out_np.copy())
            if ephemeral:
                acl.rt.free_host(host_buf)

        return outputs

def postprocess_pose(outputs, img_shape, conf_thres=0.5):
    """
    YOLOv8-pose 后处理
    输出格式: (1, 56, 8400)
      56 = 4(bbox) + 1(conf) + 51(17*3 关键点)
      8400 = 检测框数量

    返回: [{
        'box': [x1, y1, x2, y2],
        'conf': float,
        'keypoints': [[x, y, conf], ...] * 17
    }]
    """
    if not outputs or len(outputs) == 0:
        return []

    pred = outputs[0]  # shape: (56*8400,) flat
    pred = pred.reshape(56, 8400).T  # (8400, 56)

    # 提取 bbox + conf + keypoints
    boxes_xywh = pred[:, :4]  # (8400, 4)
    confs_raw = pred[:, 4]  # (8400,)
    keypoints_raw = pred[:, 5:]  # (8400, 51) = 17*3

    # 应用 sigmoid 激活到置信度 (模型输出是 logits)
    confs = 1.0 / (1.0 + np.exp(-confs_raw))

    # DEBUG: 显示最大置信度
    max_conf = confs.max()
    max_conf_raw = confs_raw.max()
    print(f"  [DEBUG] 原始最大值: {max_conf_raw:.3f}, sigmoid后: {max_conf:.3f}, 阈值: {conf_thres}")

    # 置信度过滤
    mask = confs >= conf_thres
    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    keypoints_raw = keypoints_raw[mask]

    if len(boxes_xywh) == 0:
        return []

    # 转换 xywh -> xyxy
    boxes_xyxy = np.zeros_like(boxes_xywh)
    boxes_xyxy[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

    # 缩放到原图
    h, w = img_shape[:2]
    boxes_xyxy[:, [0, 2]] *= w / 640
    boxes_xyxy[:, [1, 3]] *= h / 640

    # 解析关键点
    results = []
    for i in range(len(boxes_xyxy)):
        kpts = keypoints_raw[i].reshape(17, 3)  # (17, 3) - [x, y, conf]
        kpts[:, 0] *= w / 640
        kpts[:, 1] *= h / 640

        results.append({
            'box': boxes_xyxy[i].tolist(),
            'conf': float(confs[i]),
            'keypoints': kpts.tolist()
        })

    return results

def check_hand_safety(person_det, depth_frame, danger_distance=150):
    """
    检查手部安全：查询手腕关键点深度

    COCO 关键点索引：
      9: 左手腕, 10: 右手腕
      7: 左肘,   8: 右肘
      5: 左肩,   6: 右肩

    返回: (is_danger, min_depth_mm, wrist_positions)
    """
    keypoints = np.array(person_det['keypoints'])

    # 提取手腕关键点 (索引 9, 10)
    wrist_ids = [9, 10]
    wrists = []

    for wid in wrist_ids:
        x, y, conf = keypoints[wid]
        if conf > 0.5:  # 置信度过滤
            wrists.append((int(x), int(y), conf))

    if not wrists:
        return False, None, []

    # 查询深度
    depths = []
    for x, y, conf in wrists:
        try:
            depth_mm = depth_frame.get_distance(x, y) * 1000  # m -> mm
            if depth_mm > 0:  # 有效深度
                depths.append(depth_mm)
        except:
            pass

    if not depths:
        return False, None, wrists

    min_depth = min(depths)
    is_danger = min_depth < danger_distance

    return is_danger, min_depth, wrists

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='OM 模型路径')
    parser.add_argument('--danger-distance', type=int, default=150, help='危险距离 (mm)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='置信度阈值')
    args = parser.parse_args()

    print("=" * 60)
    print("Hand Safety Monitor - YOLOv8-pose + RealSense D435")
    print("=" * 60)

    # 1. 初始化 ACL
    res = AclLiteResource()
    res.init()

    # 2. 加载模型
    model = AclLiteModel(args.model)
    model.load()
    model.context = res.context

    # 3. 初始化 RealSense D435
    print("\n[3/4] 初始化 RealSense D435...")
    try:
        import pyrealsense2 as rs

        # 查找 RealSense 设备
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未找到 RealSense 设备")

        # 使用第一个设备的序列号
        serial = devices[0].get_info(rs.camera_info.serial_number)
        print(f"  设备序列号: {serial}")

        pipeline = rs.pipeline()
        config = rs.config()
        # 明确指定设备序列号，避免冲突
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("✓ RealSense D435 已连接")
    except Exception as e:
        print(f"✗ RealSense 初始化失败: {e}")
        print("  提示: pip install pyrealsense2")
        return

    # 4. WebUI 输出目录
    web_dir = Path.home() / 'ICT' / 'webui_http_safety'
    web_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[4/4] WebUI 输出: {web_dir}")
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

            # YOLO 推理
            img_in = cv2.resize(frame, (640, 640)).astype(np.uint8)
            outputs = model.execute(img_in)

            infer_ms = (time.time() - t0) * 1000

            # 后处理
            persons = postprocess_pose(outputs, frame.shape, args.conf_thres)

            # 安全检查
            is_danger = False
            min_depth_mm = None
            for person in persons:
                danger, depth, wrists = check_hand_safety(person, depth_frame, args.danger_distance)
                if danger:
                    is_danger = True
                    min_depth_mm = depth

                    # 绘制警告
                    for x, y, _ in wrists:
                        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                    cv2.putText(frame, f"DANGER! {depth:.0f}mm", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # FPS 计算
            now = time.time()
            fps = 1.0 / (now - last_ts) if last_ts else 0.0
            last_ts = now

            # 每秒输出一次状态
            frame_count += 1
            if frame_count % 30 == 0:
                status = "⚠ DANGER" if is_danger else "✓ SAFE"
                print(f"{status} | FPS: {fps:.1f} | 推理: {infer_ms:.1f}ms | 人数: {len(persons)}")

            # WebUI 更新（每 33ms）
            if frame_count % 1 == 0:
                # 保存图像
                cv2.imwrite(str(web_dir / 'frame.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

                # 保存状态
                state = {
                    'fps': fps,
                    'infer_ms': infer_ms,
                    'persons': len(persons),
                    'is_danger': is_danger,
                    'min_depth_mm': min_depth_mm,
                    'ts': now
                }
                with open(web_dir / 'state.json', 'w') as f:
                    json.dump(state, f)

    except KeyboardInterrupt:
        print("\n\n监控已停止")
    finally:
        pipeline.stop()
        print("✓ RealSense 已关闭")

if __name__ == '__main__':
    main()
