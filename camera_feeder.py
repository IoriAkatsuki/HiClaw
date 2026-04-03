import cv2
import time
import os

cap = None
# Try devices
for i in [4, 0, 1, 2, 3, 5]:
    try:
        c = cv2.VideoCapture(i)
        if c.isOpened():
            ret, _ = c.read()
            if ret:
                print(f"Selected device {i}")
                cap = c
                break
            c.release()
    except:
        pass

if cap is None:
    print("No valid camera found")
    exit(1)

cap.set(3, 640)
cap.set(4, 480)

tmp_file = "/home/HwHiAiUser/ICT/webui/ipc_frame.tmp.jpg"
target_file = "/tmp/ipc_frame.jpg" # Target still needs to be consistent with C++ reader

print(f"Start feeding, tmp={tmp_file}, target={target_file}", flush=True)
while True:
    ret, frame = cap.read()
    if ret:
        success = cv2.imwrite(tmp_file, frame)
        if success:
            try:
                os.rename(tmp_file, target_file)
            except OSError as e:
                print(f"Rename failed: {e}", flush=True)
        else:
            print("cv2.imwrite returned False", flush=True)
    else:
        print("Read failed", flush=True)
    time.sleep(0.03)
