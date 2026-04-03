# 相机像素 + 深度 → 振镜坐标 映射模块

本目录提供**相机坐标系 → 振镜坐标系**的标定与推理封装，用于电子元件分拣等场景。

目标场景：

- D435 输出对齐好的彩色 + 深度图；
- YOLOv8 在彩色图上识别电阻/电容等元件，得到像素框；
- 利用深度 + 标定，把像素框中心变成振镜坐标 `(X_galvo, Y_galvo)`，驱动激光/执行器。

---

## 核心思想

1. **像素 + 深度 → 相机三维坐标**  
   使用针孔模型和相机内参 `(fx, fy, cx, cy)`，把像素 `(u, v)` 与深度 `Z` 转为相机坐标系三维点：

   \[
   X = (u - c_x) / f_x \cdot Z,\quad
   Y = (v - c_y) / f_y \cdot Z,\quad
   Z = Z
   \]

2. **相机平面 → 振镜平面：单应性标定**  
   假设工作面（传送带/工装板）近似为平面，只需在平面上做 2D 映射：

   - 在不同振镜指令 `(X_galvo, Y_galvo)` 位置打点；
   - 相机拍图 + 深度，算出这些点在相机平面上的坐标 `(x_cam, y_cam)`；
   - 用 `cv2.findHomography` 求解单应性矩阵 `H`：

   \[
   \begin{bmatrix} X_g \\ Y_g \\ 1 \end{bmatrix}
   \propto
   H \cdot
   \begin{bmatrix} x_{\text{cam}} \\ y_{\text{cam}} \\ 1 \end{bmatrix}
   \]

3. **在线推理阶段**  
   - YOLO 输出像素框中心 `(u, v)`；
   - 深度图给出 `Z`；
   - 通过本模块完成：

   ```text
   (u, v, Z) → P_cam(X, Y, Z) → (x_cam, y_cam = X, Y) → (X_galvo, Y_galvo)
   ```

---

## 模块结构

- `camera_galvo_mapping.py`
  - `CameraIntrinsics`：相机内参数据类。
  - `HomographyCalib`：存放单应性矩阵 H 的数据类，支持保存/加载。
  - `pixel_depth_to_cam(u, v, depth, intr)`：像素 + 深度 → `P_cam = [X, Y, Z]`。
  - `calibrate_camera_to_galvo_homography(cam_points, galvo_points)`：根据一组对应点求 H。
  - `cam_to_galvo(x_cam, y_cam, calib)`：相机平面坐标 → 振镜平面坐标。
  - `pixel_depth_to_galvo(u, v, depth, intr, calib)`：一行封装，推理阶段直接得到 `(X_galvo, Y_galvo)`。

---

## 标定流程示例

1. **采集对应点**

   - 固定相机与工作面；
   - 让振镜在若干位置打点（或点亮光斑），记录控制指令：

     ```text
     P_galvo_i = (X_galvo_i, Y_galvo_i)
     ```

   - 用相机 + 深度获取每个点在相机侧的平面坐标，例如使用相机坐标系 X,Y：

     ```python
     from camera_galvo_mapping import CameraIntrinsics, pixel_depth_to_cam

     intr = CameraIntrinsics(fx, fy, cx, cy)
     cam_pts = []
     galvo_pts = []

     for each calibration point:
         u, v = 像素坐标（通过简单的亮点检测/质心计算得到）
         depth = 深度图中 (u, v) 周围的中位数
         X, Y, Z = pixel_depth_to_cam(u, v, depth, intr)
         cam_pts.append((X, Y))
         galvo_pts.append((X_galvo, Y_galvo))
     ```

2. **估计单应性矩阵 H**

   ```python
   from camera_galvo_mapping import calibrate_camera_to_galvo_homography

   calib = calibrate_camera_to_galvo_homography(cam_pts, galvo_pts)
   calib.save("cam_to_galvo_H.npy")
   ```

3. **在线推理时使用**

   ```python
   from camera_galvo_mapping import (
       CameraIntrinsics,
       HomographyCalib,
       pixel_depth_to_galvo,
   )

   intr = CameraIntrinsics(fx, fy, cx, cy)
   calib = HomographyCalib.load("cam_to_galvo_H.npy")

   # 假设 YOLO 检测到一个电阻，框中心像素为 (u, v)
   # depth_map 为对齐到彩色图的深度图
   depth_roi = depth_map[v-2:v+3, u-2:u+3]
   depth = float(np.median(depth_roi[depth_roi > 0]))

   X_galvo, Y_galvo = pixel_depth_to_galvo(u, v, depth, intr, calib)

   # 后续：将 (X_galvo, Y_galvo) 传给振镜/执行器控制模块
   ```

---

## 与板卡项目的集成建议

1. **在 PC 侧完成标定**  
   - 利用 Python + D435 + 振镜控制，运行一次标定脚本，得到 `cam_to_galvo_H.npy`。

2. **将标定结果拷贝到板卡**  
   - 如：拷贝到 `~/ICT/edge/common/calibration/cam_to_galvo_H.npy`。

3. **在板卡端 C++/Python 推理代码中集成**  
   - 检测到目标后，将像素 + 深度转为相机坐标系 X,Y；
   - 用同样的 H 做一次齐次变换，得到振镜坐标；
   - 保持与当前激光控制接口的单位/比例一致即可。

---

## 注意事项

- 标定点应尽量覆盖工作区域（四角 + 中心 + 中间若干点），避免只在一小块区域采样导致外推误差大。
- 如果工作面不完全平整，单纯 2D 单应性会有误差，后续可升级为 3D 刚体 + 平面拟合方案。
- 深度噪声较大时，建议对 ROI 使用中值或均值滤波，并做异常值剔除。 

