#!/usr/bin/env python3
"""
Board Test 3: Visual Protocol Verification with Camera
=======================================================
Run on board: python3 -u tests/board_test_visual_verify.py

Uses RealSense camera to capture frames while sending galvo commands.
Detects laser spot position to verify commands are actually working.
Saves images to /tmp/galvo_test_*.jpg for review.

This is the automated version of Test 1 — no human observation needed.
"""
import serial
import time
import sys
import os

PORT = "/dev/ttyUSB0"
BAUD = 115200
OUTPUT_DIR = "/tmp/galvo_test"

def send(ser, cmd):
    ser.write(cmd.encode('ascii'))
    ser.flush()

def capture_frame(pipeline):
    """Capture one frame from RealSense."""
    import pyrealsense2 as rs
    import numpy as np
    frames = pipeline.wait_for_frames(timeout_ms=2000)
    color_frame = frames.get_color_frame()
    return np.asanyarray(color_frame.get_data())

def detect_red_spot(frame):
    """Detect red laser spot in frame, return (x, y) or None."""
    import cv2
    import numpy as np
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red laser: high saturation, high value
    mask1 = cv2.inRange(hsv, np.array([0, 100, 200]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([160, 100, 200]), np.array([180, 255, 255]))
    # Also check for very bright white/saturated spots (laser overexposure)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    mask = mask1 | mask2 | bright_mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 5:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def test_visual():
    import cv2
    import pyrealsense2 as rs

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Init RealSense
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(config)
    time.sleep(1)  # warm up

    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(0.5)

    print("=" * 60)
    print("TEST 3: Visual Protocol Verification")
    print("=" * 60)

    results = []

    # Test cases: (label, commands_to_send, expected_behavior)
    tests = [
        ("baseline_no_cmd", [], "No laser movement"),
        ("new_fmt_0R", ["0R,0,0,3000,3000;U\n"], "Index-first format — expect NO response"),
        ("old_fmt_R0_center", ["R00,0,3000,3000\n", "U\n"], "Command-first rect at center"),
        ("old_fmt_C0_center", ["C00,0,2000\n", "U\n"], "Command-first circle at center"),
        ("old_fmt_R0_offset", ["R05000,5000,2000,2000\n", "U\n"], "Rect at +5000,+5000"),
        ("u_only_clear", ["U\n"], "U-only — should stop drawing"),
    ]

    for label, cmds, desc in tests:
        print(f"\n--- {label}: {desc} ---")

        # Flush frames
        for _ in range(5):
            capture_frame(pipe)

        # Send commands
        for cmd in cmds:
            send(ser, cmd)
            print(f"  Sent: {repr(cmd)}")
            time.sleep(0.05)

        # Wait for galvo to respond
        time.sleep(1.0)

        # Capture multiple frames
        spots = []
        for i in range(10):
            frame = capture_frame(pipe)
            spot = detect_red_spot(frame)
            spots.append(spot)
            if i == 5:
                path = os.path.join(OUTPUT_DIR, f"{label}.jpg")
                cv2.imwrite(path, frame)
                print(f"  Saved: {path}")

        detected = [s for s in spots if s is not None]
        if detected:
            avg_x = sum(s[0] for s in detected) / len(detected)
            avg_y = sum(s[1] for s in detected) / len(detected)
            print(f"  Laser spot detected: avg=({avg_x:.0f}, {avg_y:.0f}), {len(detected)}/10 frames")
            results.append((label, True, f"({avg_x:.0f}, {avg_y:.0f})"))
        else:
            print(f"  No laser spot detected in any frame")
            results.append((label, False, "none"))

    # Cleanup
    send(ser, "U\n")
    ser.close()
    pipe.stop()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Test':<25} {'Spot?':<8} {'Position'}")
    print("-" * 50)
    for label, detected, pos in results:
        print(f"{label:<25} {'YES' if detected else 'NO':<8} {pos}")
    print("=" * 60)
    print(f"Images saved to: {OUTPUT_DIR}/")

if __name__ == "__main__":
    try:
        test_visual()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
