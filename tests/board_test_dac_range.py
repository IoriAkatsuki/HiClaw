#!/usr/bin/env python3
"""
Board Test 2: DAC Range Sweep
==============================
Run on board: python3 -u tests/board_test_dac_range.py

Sweeps galvo positions from center outward to find physical limits.
Observes where the laser spot stops moving or hits mechanical limits.
Uses CONFIRMED firmware format: C<idx><x>,<y>,<r>

Safety: starts small, increases gradually. Ctrl+C to abort.
"""
import serial
import time
import sys

PORT = "/dev/ttyUSB0"
BAUD = 115200

def send(ser, cmd):
    ser.write(cmd.encode('ascii'))
    ser.flush()

def goto(ser, x, y):
    """Move galvo to position using a small circle marker."""
    send(ser, f"C0{x},{y},200\n")
    time.sleep(0.02)
    send(ser, "U\n")

def test_dac_range():
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(0.5)
    print("=" * 60)
    print("TEST 2: DAC Range Sweep")
    print("=" * 60)
    print("Observe the laser spot. Note where it stops moving.")
    print("Press Ctrl+C to abort at any time.\n")

    # Phase A: X-axis sweep (Y=0)
    print("[A] X-axis sweep (Y=0, positive direction)")
    test_values = list(range(0, 32001, 2000))
    for x in test_values:
        goto(ser, x, 0)
        print(f"  X={x:+6d}, Y=0")
        time.sleep(1.0)

    print("\n[A'] X-axis sweep (Y=0, negative direction)")
    for x in test_values:
        goto(ser, -x, 0)
        print(f"  X={-x:+6d}, Y=0")
        time.sleep(1.0)

    # Phase B: Y-axis sweep (X=0)
    print("\n[B] Y-axis sweep (X=0, positive direction)")
    for y in test_values:
        goto(ser, 0, y)
        print(f"  X=0, Y={y:+6d}")
        time.sleep(1.0)

    print("\n[B'] Y-axis sweep (X=0, negative direction)")
    for y in test_values:
        goto(ser, 0, -y)
        print(f"  X=0, Y={-y:+6d}")
        time.sleep(1.0)

    # Phase C: Diagonal sweep
    print("\n[C] Diagonal sweep (X=Y)")
    for v in test_values:
        goto(ser, v, v)
        print(f"  X={v:+6d}, Y={v:+6d}")
        time.sleep(1.0)

    # Return to center
    goto(ser, 0, 0)
    print("\n  Returned to center (0, 0)")

    # Clear
    send(ser, "U\n")
    ser.close()
    print("\n" + "=" * 60)
    print("TEST 2 COMPLETE")
    print("Record the approximate values where laser stopped moving.")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_dac_range()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        ser = serial.Serial(PORT, BAUD, timeout=1)
        send(ser, "U\n")
        ser.close()
