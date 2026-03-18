#!/usr/bin/env python3
"""
Board Test 1: Protocol Format Verification
===========================================
Run on board: python3 -u tests/board_test_protocol.py

Tests which UART command format the STM32 firmware actually accepts.
Requires: galvo + STM32 connected via /dev/ttyUSB0, laser visible.

Expected behavior per firmware main.c analysis:
  - 固件通过 UART6 空闲中断回调解析整包字符串
  - 使用 ';' 分割多个 token
  - token 格式为: <idx><R/C>,<args>，最后用 U; 提交
  - 例如: 0C,1000,2000,500;1R,0,0,3000,3000;U;
"""
import serial
import time
import sys

PORT = "/dev/ttyUSB0"
BAUD = 115200

def send(ser, cmd, label=""):
    """Send command and print what was sent."""
    data = cmd.encode('ascii')
    ser.write(data)
    ser.flush()
    print(f"  [{label}] Sent {len(data)} bytes: {repr(cmd)}")

def test_protocol():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(0.5)  # let STM32 settle after connection
    print("=" * 60)
    print("TEST 1: Protocol Format Verification")
    print("=" * 60)

    # --- Test A: 新正式格式（索引在前，分号分隔）— expected to WORK ---
    print("\n[A] New format: '0R,0,0,4000,4000;U;'")
    print("    Expected: Laser draws rectangle at center")
    send(ser, "0R,0,0,4000,4000;U;", "new-format")
    time.sleep(3)
    input_or_wait("    >> Did the laser draw a rectangle at center? (y/n): ")

    # --- Test B: 旧格式（命令在前）— expected to FAIL ---
    print("\n[B] Old format: 'R00,0,4000,4000\\n' then 'U\\n'")
    print("    Expected: NO laser movement")
    send(ser, "R00,0,4000,4000\n", "old-cmd-R")
    time.sleep(0.05)
    send(ser, "U\n", "old-cmd-U")
    time.sleep(3)
    input_or_wait("    >> Did the laser stay unchanged? (y/n): ")

    # --- Test C: 单圆命令 ---
    print("\n[C] Circle token: '0C,0,0,2000;U;'")
    print("    Expected: Laser draws circle at center")
    send(ser, "0C,0,0,2000;U;", "new-cmd-C")
    time.sleep(3)
    input_or_wait("    >> Did the laser draw a circle at center? (y/n): ")

    # --- Test D: Multiple tasks ---
    print("\n[D] Two tasks in one packet: 0R + 1C + U")
    send(ser, "0R,0,0,3000,3000;1C,5000,5000,1000;U;", "packet-R0-C1-U")
    time.sleep(3)
    input_or_wait("    >> Did the laser draw a rectangle + circle? (y/n): ")

    # --- Test E: Clear (U-only) ---
    print("\n[E] U-only (clear): 'U;'")
    print("    Expected: Laser stops drawing (all tasks become NONE)")
    send(ser, "U;", "cmd-U-only")
    time.sleep(3)
    input_or_wait("    >> Did the laser stop? Is laser dot still visible? (y/n/describe): ")

    ser.close()
    print("\n" + "=" * 60)
    print("TEST 1 COMPLETE")
    print("=" * 60)

def input_or_wait(prompt):
    """In non-interactive mode, just wait and print the prompt."""
    if sys.stdin.isatty():
        return input(prompt)
    else:
        print(prompt + "(non-interactive, skipping)")
        return ""

if __name__ == "__main__":
    test_protocol()
