#!/usr/bin/env python3
"""
Board Test 1: Protocol Format Verification
===========================================
Run on board: python3 -u tests/board_test_protocol.py

Tests which UART command format the STM32 firmware actually accepts.
Requires: galvo + STM32 connected via /dev/ttyUSB0, laser visible.

Expected behavior per firmware main.c analysis:
  - Firmware ISR checks uart1_rx_buf[0] for command type ('R', 'C', 'U')
  - Index is uart1_rx_buf[1] as ASCII digit
  - Format: R<idx><x>,<y>,<w>,<h>  or  C<idx><x>,<y>,<r>
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

    # --- Test A: New format (index-first) — expected to FAIL ---
    print("\n[A] Index-first format: '0R,0,0,2000,2000;U\\n'")
    print("    Expected: NO laser movement (firmware sees '0', not 'R')")
    send(ser, "0R,0,0,2000,2000;U\n", "new-format")
    time.sleep(3)
    input_or_wait("    >> Did the laser draw a rectangle at center? (y/n): ")

    # --- Test B: Firmware format (command-first) — expected to WORK ---
    print("\n[B] Command-first format: 'R00,0,4000,4000\\n' then 'U\\n'")
    print("    Expected: Laser draws rectangle at center")
    send(ser, "R00,0,4000,4000\n", "cmd-R")
    time.sleep(0.05)
    send(ser, "U\n", "cmd-U")
    time.sleep(3)
    input_or_wait("    >> Did the laser draw a rectangle at center? (y/n): ")

    # --- Test C: Circle command-first ---
    print("\n[C] Circle command-first: 'C00,0,2000\\n' then 'U\\n'")
    print("    Expected: Laser draws circle at center")
    send(ser, "C00,0,2000\n", "cmd-C")
    time.sleep(0.05)
    send(ser, "U\n", "cmd-U")
    time.sleep(3)
    input_or_wait("    >> Did the laser draw a circle at center? (y/n): ")

    # --- Test D: Multiple tasks ---
    print("\n[D] Two tasks: R0 + C1, then U")
    send(ser, "R00,0,3000,3000\n", "cmd-R0")
    time.sleep(0.05)
    send(ser, "C15000,5000,1000\n", "cmd-C1")
    time.sleep(0.05)
    send(ser, "U\n", "cmd-U")
    time.sleep(3)
    input_or_wait("    >> Did the laser draw a rectangle + circle? (y/n): ")

    # --- Test E: Clear (U-only) ---
    print("\n[E] U-only (clear): 'U\\n'")
    print("    Expected: Laser stops drawing (all tasks become NONE)")
    send(ser, "U\n", "cmd-U-only")
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
