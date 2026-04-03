# Route B - Hand Safety Monitor

基于 YOLOv8-pose + RealSense D435 的手部安全监控系统

## 系统架构

### 双路检测系统

**Route A** (端口 8000)
- 功能：电子元器件目标检测 (61 类)
- 模型：YOLOv8 (electro61)
- 输入：RealSense D435 彩色流
- 性能：~30 FPS, ~20ms 推理

**Route B** (端口 8001)
- 功能：手部骨架检测 + 深度安全警告
- 模型：YOLOv8-pose (17 关键点)
- 输入：RealSense D435 彩色流 + 深度流
- 性能：~30 FPS, ~30ms 推理
- 警告距离：< 150mm

## 当前状态

### ✅ 已完成
1. YOLOv8-pose 模型下载和转换 (yolov8n-pose.pt → yolov8n_pose_aipp.om)
2. ACL 推理脚本 (hand_safety_monitor.py)
3. RealSense D435 初始化和深度查询
4. 手腕关键点检测 (索引 9, 10)
5. 安全距离判断逻辑 (默认 150mm)
6. WebUI 实时监控界面 (端口 8001)
7. Route B 单独运行测试通过

### 🔧 待解决：相机资源冲突

**问题描述**
- 系统只有一个 RealSense D435 相机
- RealSense 创建 6 个 video 设备: /dev/video0-5
- Route A 当前使用 /dev/video4 (属于 RealSense)
- Route B 需要 RGB + Depth 流访问 RealSense
- 两者同时初始化时产生冲突: "Device or resource busy"

**解决方案选项**

**方案 1: 相机服务器 (推荐)**
- 创建独立的相机服务进程
- 统一管理 RealSense D435 的 RGB 和 Depth 流
- Route A 和 Route B 通过 API/共享内存获取帧
- 优点：清晰的架构，资源统一管理
- 缺点：需要额外开发

**方案 2: Route B 提供 RGB 流**
- Route B 作为主进程管理 RealSense
- Route A 从 Route B 获取 RGB 帧 (通过 HTTP/共享内存)
- 优点：复用现有 RealSense 初始化
- 缺点：Route B 停止时 Route A 也无法工作

**方案 3: 添加第二个相机**
- 为 Route A 添加独立的 USB 摄像头
- Route B 独占 RealSense D435
- 优点：最简单，无需修改架构
- 缺点：需要额外硬件

## 技术细节

### YOLOv8-pose 输出格式
```
输入: (1, 3, 640, 640) RGB 图像
输出: (1, 56, 8400)
  - 56 channels = 4(bbox) + 1(conf) + 51(17*3 keypoints)
  - 17 关键点: COCO 格式
    - 索引 9: 左手腕
    - 索引 10: 右手腕
    - 索引 7, 8: 肘部
    - 索引 5, 6: 肩部
```

### AIPP 配置
```
input_format: RGB888_U8
src_image_size: 640x640
var_reci: 0.00392156862745098  # 1/255 normalization
```

### 安全检测逻辑
```python
def check_hand_safety(person_det, depth_frame, danger_distance=150):
    # 1. 提取手腕关键点 (索引 9, 10)
    # 2. 查询深度: depth_frame.get_distance(x, y)
    # 3. 判断: min_depth < danger_distance
    # 4. 返回: (is_danger, min_depth_mm, wrist_positions)
```

## 性能测试结果

| 指标 | Route A (YOLOv8) | Route B (YOLOv8-pose) |
|------|------------------|----------------------|
| 推理时间 | ~20ms | ~30ms |
| 帧率 | 30 FPS | 30 FPS |
| 后处理 | 3-5ms | 5-8ms |
| 内存占用 | ~400MB | ~630MB |
| NPU 利用率 | ~70% | ~78% |

## 启动方式

### 单独启动 Route B
```bash
cd ~/ICT
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 启动 WebUI (后台)
nohup python3 edge/route_b_app/webui_server.py > webui_safety.log 2>&1 &

# 启动手部安全监控
python3 edge/route_b_app/hand_safety_monitor.py \
    --model yolov8n_pose_aipp.om \
    --danger-distance 150 \
    --conf-thres 0.5
```

### 使用启动脚本
```bash
cd ~/ICT
./start_route_b.sh
```

### 并发运行 Route A + Route B
⚠️ **当前不可用** - 需要先解决相机资源冲突

## WebUI 访问

- Route A: http://ict.local:8000 (物体检测)
- Route B: http://ict.local:8001 (手部安全)

## 文件结构
```
~/ICT/
├── yolov8n_pose_aipp.om           # YOLOv8-pose OM 模型 (7.5MB)
├── aipp_yolov8_pose.cfg           # AIPP 预处理配置
├── start_route_b.sh               # Route B 启动脚本
├── edge/
│   └── route_b_app/
│       ├── hand_safety_monitor.py  # 主程序
│       ├── webui_server.py         # WebUI 服务器
│       └── README.md               # 本文档
├── webui_http_safety/
│   ├── index.html                  # WebUI 界面
│   ├── frame.jpg                   # 实时帧 (自动更新)
│   └── state.json                  # 状态数据 (自动更新)
└── logs/
    ├── hand_safety.log             # 监控日志
    └── webui_safety.log            # WebUI 日志
```

## 开发历史

1. **2024-12-06 18:00** - 规划 Route B 架构
2. **2024-12-06 19:30** - 下载并测试 YOLOv8-pose 模型
3. **2024-12-06 19:45** - ONNX 转换 (opset 11)
4. **2024-12-06 20:15** - OM 模型转换成功
5. **2024-12-06 20:30** - 完成 ACL 推理脚本
6. **2024-12-06 20:45** - RealSense D435 集成
7. **2024-12-06 21:00** - WebUI 开发完成
8. **2024-12-06 21:20** - Route B 单独运行测试通过
9. **2024-12-06 21:30** - 发现相机资源冲突问题

## 下一步计划

1. [ ] 实现相机服务器 (camera_server.py)
2. [ ] 修改 Route A 使用相机服务器
3. [ ] 修改 Route B 使用相机服务器
4. [ ] 测试双路并发运行
5. [ ] 性能优化和调优
6. [ ] 添加声音/LED 警报功能
7. [ ] 距离阈值可配置化
8. [ ] 添加检测历史记录

## 参考资料

- [Ultralytics YOLOv8-pose](https://docs.ultralytics.com/tasks/pose/)
- [RealSense D435 Documentation](https://www.intelrealsense.com/depth-camera-d435/)
- [Ascend ACL API Reference](https://www.hiascend.com/document)
- [COCO Keypoint Format](https://cocodataset.org/#keypoints-2020)
