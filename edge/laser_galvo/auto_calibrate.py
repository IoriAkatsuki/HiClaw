#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""自动校准脚本（稳定性优先，无人值守默认开启）。"""

import argparse
import glob
import json
import time
from pathlib import Path

import cv2

from calibrate_galvo import GalvoCalibrator


def find_serial_port(preferred_port=None):
    """自动查找串口设备；有多个时默认选择第一个。"""
    patterns = ['/dev/ttyUSB*', '/dev/ttyACM*']
    ports = []
    for pattern in patterns:
        ports.extend(glob.glob(pattern))

    ports = sorted(set(ports))

    if preferred_port:
        if preferred_port in ports or Path(preferred_port).exists():
            print(f"✓ 使用指定串口: {preferred_port}")
            return preferred_port
        print(f"⚠ 指定串口不存在: {preferred_port}，改为自动探测")

    if not ports:
        print('✗ 未找到串口设备，请检查 STM32 连接')
        return None

    if len(ports) > 1:
        print(f"⚠ 检测到多个串口，默认使用: {ports[0]}")
        for p in ports:
            print(f"  - {p}")
    else:
        print(f"✓ 自动检测到串口: {ports[0]}")

    return ports[0]


def test_laser_control(serial_port, baudrate=115200, headless=True):
    """测试激光控制命令是否可发送。"""
    print('\n=== 测试激光控制链路 ===')
    calibrator = GalvoCalibrator(serial_port=serial_port, baudrate=baudrate)

    if not calibrator.connect_serial():
        return False

    try:
        calibrator.send_galvo_position(0, 0)
        time.sleep(0.2)
        calibrator.enable_laser(False)
        time.sleep(0.2)

        if headless:
            print('✓ headless模式：命令发送成功，默认判定链路可用')
            return True

        response = input('激光是否有明显亮灭变化？[y/N]: ').strip().lower()
        if response == 'y':
            print('✓ 激光控制正常')
            return True

        print('✗ 激光控制异常（命令可发送但未观察到响应）')
        return False
    except Exception as e:
        print(f'✗ 激光控制测试失败: {e}')
        return False
    finally:
        if calibrator.ser is not None:
            calibrator.ser.close()


def check_camera_available(camera_id=0, headless=True):
    """检查摄像头可用性。"""
    print('\n=== 检查相机可用性 ===')
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'✗ 无法打开摄像头 ID={camera_id}')
        return False

    frames = []
    for _ in range(15):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
        time.sleep(0.01)

    cap.release()

    if not frames:
        print('✗ 摄像头打开成功但未读取到有效帧')
        return False

    if headless:
        brightness = sum(float(f.mean()) for f in frames) / len(frames)
        print(f'✓ 相机可用，平均亮度: {brightness:.1f}')
    else:
        print('✓ 相机可用')

    return True


def build_retry_config(attempt_index, base_capture_attempts, base_settle_ms, retry_profile):
    """按尝试次数生成降级策略。"""
    if retry_profile != 'safe_then_relax':
        raise ValueError(f'不支持的重试策略: {retry_profile}')

    profiles = [
        {
            'name': 'strict',
            'capture_attempts_per_point': base_capture_attempts,
            'settle_ms': base_settle_ms,
            'detector_profile': {
                'hsv_red_low_1': [0, 65, 70],
                'hsv_red_high_1': [10, 255, 255],
                'hsv_red_low_2': [160, 65, 70],
                'hsv_red_high_2': [180, 255, 255],
                'diff_thresh': 30,
                'open_kernel': 3,
                'area_min': 10,
                'area_max': 900,
                'min_circularity': 0.18,
                'min_peak_v': 150,
                'min_inlier_radius_px': 2.5,
                'ransac_reproj_threshold': 5.0,
            },
        },
        {
            'name': 'relaxed_1',
            'capture_attempts_per_point': base_capture_attempts + 2,
            'settle_ms': base_settle_ms + 100,
            'detector_profile': {
                'hsv_red_low_1': [0, 55, 60],
                'hsv_red_high_1': [10, 255, 255],
                'hsv_red_low_2': [160, 55, 60],
                'hsv_red_high_2': [180, 255, 255],
                'diff_thresh': 24,
                'open_kernel': 3,
                'area_min': 8,
                'area_max': 1200,
                'min_circularity': 0.15,
                'min_peak_v': 130,
                'min_inlier_radius_px': 3.0,
                'ransac_reproj_threshold': 6.0,
            },
        },
        {
            'name': 'relaxed_2',
            'capture_attempts_per_point': base_capture_attempts + 4,
            'settle_ms': base_settle_ms + 200,
            'detector_profile': {
                'hsv_red_low_1': [0, 50, 55],
                'hsv_red_high_1': [10, 255, 255],
                'hsv_red_low_2': [160, 50, 55],
                'hsv_red_high_2': [180, 255, 255],
                'diff_thresh': 18,
                'open_kernel': 3,
                'area_min': 6,
                'area_max': 1500,
                'min_circularity': 0.12,
                'min_peak_v': 110,
                'min_inlier_radius_px': 3.5,
                'ransac_reproj_threshold': 7.0,
            },
        },
    ]

    return profiles[min(attempt_index, len(profiles) - 1)]


def attempt_diag_path(base_path, attempt):
    p = Path(base_path)
    suffix = p.suffix if p.suffix else '.json'
    stem = p.stem if p.suffix else p.name
    return str(p.with_name(f"{stem}_attempt{attempt}{suffix}"))


def run_calibration_with_retry(
    serial_port,
    baudrate,
    camera_id,
    output_file,
    diagnostic_file,
    max_retries,
    retry_profile,
    headless,
    min_valid_points,
    mean_error_thres,
    max_error_thres,
    capture_attempts_per_point,
    settle_ms,
):
    """运行校准，支持自动降级重试。"""
    attempt_reports = []

    for attempt in range(1, max_retries + 1):
        config = build_retry_config(attempt - 1, capture_attempts_per_point, settle_ms, retry_profile)

        print('\n' + '=' * 60)
        print(f"校准尝试 {attempt}/{max_retries} | 策略: {config['name']}")
        print('=' * 60)

        calibrator = GalvoCalibrator(
            serial_port=serial_port,
            baudrate=baudrate,
            laser_color='red',
            capture_attempts_per_point=config['capture_attempts_per_point'],
            settle_ms=config['settle_ms'],
            min_valid_points=min_valid_points,
            mean_error_thres=mean_error_thres,
            max_error_thres=max_error_thres,
            diagnostic_file=diagnostic_file,
            detector_profile=config['detector_profile'],
        )

        if not calibrator.connect_serial():
            report = {
                'attempt': attempt,
                'profile': config['name'],
                'success': False,
                'reason': '串口连接失败',
            }
            attempt_reports.append(report)
            continue

        success = calibrator.calibrate_with_camera(camera_id=camera_id, debug=not headless)

        this_attempt_diag = attempt_diag_path(diagnostic_file, attempt)
        calibrator.save_diagnostic(
            this_attempt_diag,
            extra_context={
                'attempt': attempt,
                'retry_profile': retry_profile,
                'profile_name': config['name'],
            },
        )

        report = {
            'attempt': attempt,
            'profile': config['name'],
            'success': bool(success),
            'reason': calibrator.last_failure_reason,
            'quality': calibrator.quality_metrics,
            'diagnostic_file': this_attempt_diag,
        }
        attempt_reports.append(report)

        if success:
            calibrator.save_calibration(output_file)
            return True, attempt_reports

        print(f"✗ 本次失败: {calibrator.last_failure_reason}")

    return False, attempt_reports


def main():
    parser = argparse.ArgumentParser(description='激光振镜自动校准（无人值守）')
    parser.add_argument('--serial-port', help='串口设备（为空则自动探测）')
    parser.add_argument('--baudrate', type=int, default=115200, help='波特率')
    parser.add_argument('--camera-id', type=int, default=0, help='摄像头 ID')

    parser.add_argument('--output-dir', default='~/ICT', help='输出目录')
    parser.add_argument('--output-file', default='galvo_calibration.yaml', help='标定文件名或绝对路径')
    parser.add_argument('--diagnostic-file', default='calibration_diagnostic.json', help='诊断文件名或绝对路径')

    parser.add_argument('--max-retries', type=int, default=3, help='最大重试次数')
    parser.add_argument('--retry-profile', choices=['safe_then_relax'], default='safe_then_relax', help='重试策略')

    parser.add_argument('--capture-attempts-per-point', type=int, default=12, help='每个点采样帧数')
    parser.add_argument('--settle-ms', type=int, default=500, help='每个点稳定等待时间（ms）')

    parser.add_argument('--min-valid-points', type=int, default=7, help='最小有效点数')
    parser.add_argument('--mean-error-thres', type=float, default=300.0, help='平均误差阈值')
    parser.add_argument('--max-error-thres', type=float, default=700.0, help='最大误差阈值')

    parser.add_argument('--headless', dest='headless', action='store_true', default=True, help='启用无人值守模式（默认）')
    parser.add_argument('--interactive', dest='headless', action='store_false', help='启用交互模式')

    args = parser.parse_args()

    print('=' * 60)
    print('   激光振镜自动校准 v2.0 (稳定性优先)')
    print('=' * 60)

    serial_port = find_serial_port(preferred_port=args.serial_port)
    if serial_port is None:
        return 1

    if not test_laser_control(serial_port, args.baudrate, headless=args.headless):
        print('✗ 激光链路测试失败，终止')
        return 1

    if not check_camera_available(args.camera_id, headless=args.headless):
        print('✗ 相机检查失败，终止')
        return 1

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(args.output_file)
    if not output_path.is_absolute():
        output_path = output_dir / output_path

    diagnostic_path = Path(args.diagnostic_file)
    if not diagnostic_path.is_absolute():
        diagnostic_path = output_dir / diagnostic_path

    print('\n=== 开始自动校准 ===')
    print(f'输出文件: {output_path}')
    print(f'诊断文件: {diagnostic_path}')

    success, reports = run_calibration_with_retry(
        serial_port=serial_port,
        baudrate=args.baudrate,
        camera_id=args.camera_id,
        output_file=str(output_path),
        diagnostic_file=str(diagnostic_path),
        max_retries=args.max_retries,
        retry_profile=args.retry_profile,
        headless=args.headless,
        min_valid_points=args.min_valid_points,
        mean_error_thres=args.mean_error_thres,
        max_error_thres=args.max_error_thres,
        capture_attempts_per_point=args.capture_attempts_per_point,
        settle_ms=args.settle_ms,
    )

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'success': bool(success),
        'output_file': str(output_path),
        'attempt_reports': reports,
        'args': vars(args),
    }
    with open(diagnostic_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if success:
        print('\n' + '=' * 60)
        print('✓ 校准成功')
        print(f'标定文件: {output_path}')
        print(f'诊断汇总: {diagnostic_path}')
        print('=' * 60)
        return 0

    print('\n' + '=' * 60)
    print('✗ 校准失败')
    print(f'诊断汇总: {diagnostic_path}')
    print('建议: 检查激光亮度、相机曝光和工作区域反光')
    print('=' * 60)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
