# CV 模块说明（本机 YOLOv8 测试与开发）

本目录用于在 **上位机（PC）端** 做视觉逻辑的快速验证，不依赖板卡 NPU 和 `.om` 模型，直接用 Ultralytics YOLOv8（PyTorch）完成：

- 调用本地摄像头实时检测（查看 FPS、画框效果）  
- 对离线图片/文件夹做批量检测（例如 ElectroCom61 数据集）  
- 后续可以在此基础上逐步接入深度、坐标转换等逻辑，再移植到板卡端

> 注意：这里的代码只跑在 PC 上，用的是 `best.pt` 权重；板卡上仍然使用你转换好的 `.om` 模型。

---

## 1. 环境依赖

在你的 PC 上（已经用来训练 YOLOv8），确保安装了：

- Python ≥ 3.8  
- `ultralytics`  
- `opencv-python`  

如需补装：

```bash
python3 -m pip install --upgrade pip
python3 -m pip install ultralytics opencv-python
```

---

## 2. 实时摄像头检测：`yolov8_cam.py`

脚本路径：`CV/yolov8_cam.py`

默认使用你训练好的 ElectroCom61 模型：

- 模型权重：`runs/detect/train_electro61/weights/best.pt`
- 输入尺寸：`640`
- 置信度阈值：`0.25`

### 启动方式

```bash
cd ~/Documents/ICT
python3 CV/yolov8_cam.py
```

可选参数：

```bash
python3 CV/yolov8_cam.py \
  --model runs/detect/train_electro61/weights/best.pt \
  --device 0 \
  --imgsz 640 \
  --conf 0.25 \
  --camera 0
```

- `--camera`：摄像头编号（默认 0，可换成 1、2 或 RTSP/USB 摄像头地址）  
- `--device`：`0` 表示用第 0 块 GPU，`cpu` 表示强制 CPU  

窗口中会显示实时检测结果与简单 FPS 指标，按 `q` 退出。

---

## 3. 离线图片/文件夹检测：`yolov8_eval.py`

脚本路径：`CV/yolov8_eval.py`

示例：对 ElectroCom61 验证集图片做一次可视化检测：

```bash
cd ~/Documents/ICT
python3 CV/yolov8_eval.py \
  --model runs/detect/train_electro61/weights/best.pt \
  --source "/home/oasis/Documents/datasets/ElectroCom61/ElectroCom61 A Multiclass Dataset for Detection of Electronic Components/ElectroCom-61_v2/valid/images" \
  --conf 0.25 \
  --save-dir CV/vis_valid
```

生成结果将保存到 `--save-dir` 指定的文件夹中。

---

## 4. 与标定 / 振镜的后续对接

后续如果要在 PC 端验证「检测 → 深度 → 振镜坐标」的完整链路，可以在本目录的脚本中：

1. 在获取到每个检测框后，取框中心 `(u, v)`；  
2. 从 D435 深度图中取对应深度 `Z`；  
3. 调用 `edge/common/calibration/camera_galvo_mapping.py` 里的：

   ```python
   from edge.common.calibration.camera_galvo_mapping import (
       CameraIntrinsics, HomographyCalib, pixel_depth_to_galvo,
   )
   ```

4. 得到 `(X_galvo, Y_galvo)`，再通过网络/串口发送给板卡或振镜控制模块。

目前先把 **检测部分** 独立出来，便于你在 PC 上快速调试视觉逻辑。

