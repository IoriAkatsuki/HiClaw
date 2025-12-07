#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealSense D435 简单测试
"""
import pyrealsense2 as rs
import numpy as np

def test_realsense():
    """测试 RealSense 相机连接"""
    print("=" * 60)
    print("RealSense D435 连接测试")
    print("=" * 60)

    # 列出设备
    ctx = rs.context()
    devices = ctx.query_devices()
    print(f"\n检测到 {len(devices)} 个 RealSense 设备:")

    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev.get_info(rs.camera_info.name)}")
        print(f"      序列号: {dev.get_info(rs.camera_info.serial_number)}")
        print(f"      固件版本: {dev.get_info(rs.camera_info.firmware_version)}")

    if len(devices) == 0:
        print("\n✗ 未检测到 RealSense 设备")
        return False

    # 尝试初始化
    print("\n尝试初始化相机...")
    try:
        pipeline = rs.pipeline()
        config = rs.config()

        # 仅启用深度流（不启用彩色流，避免冲突）
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        print("  配置: 640x480 @ 30fps (仅深度)")
        profile = pipeline.start(config)
        print("✓ 相机初始化成功")

        # 读取一帧测试
        print("\n读取测试帧...")
        frames = pipeline.wait_for_frames(timeout_ms=5000)
        depth_frame = frames.get_depth_frame()

        if depth_frame:
            w = depth_frame.get_width()
            h = depth_frame.get_height()
            print(f"✓ 深度帧读取成功: {w}x{h}")

            # 测试深度查询
            depth_mm = depth_frame.get_distance(320, 240) * 1000
            print(f"  中心点深度: {depth_mm:.0f} mm")

        pipeline.stop()
        print("\n✓ 测试完成，相机工作正常")
        return True

    except Exception as e:
        print(f"\n✗ 初始化失败: {e}")
        return False

if __name__ == "__main__":
    success = test_realsense()
    exit(0 if success else 1)
