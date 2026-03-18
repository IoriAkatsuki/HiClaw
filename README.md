# Laser-AR Sorting on MindSpore & Orange Pi

基于 MindSpore 与香橙派 AI Pro 的激光可视分拣项目，集成物体检测、手部安全监控和激光振镜标注功能。

## 项目概述

本项目实现了三个主要应用场景：
- **Route A**: 电子元器件检测分拣（YOLO + 激光振镜）
- **Route B**: 手部安全监控（YOLOv8-Pose / MediaPipe）
- **Unified App**: 统一检测系统（物体检测 + 手部安全 + 激光标注）

当前正式部署主线说明：
- **训练源模型**：`2026_3_12/runs/train/yolo26n_aug_full_8419_gpu/weights/best.pt`
- **正式部署模型**：`models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om`
- **正式运行配置**：`config/yolo26_6cls.yaml`
- **注意**：项目默认部署到华为 Ascend 310B，正式启动默认走 `OM`；`.pt/.onnx` 仅作为显式备用输入格式

## 振镜代码分层

当前仓库已经将振镜工作代码明确分为两层：

- **上位机代码**：`edge/laser_galvo/`、`edge/unified_app/`
- **单片机板卡驱动**：`mirror/stm32f401ccu6_dac8563/`

建议先阅读 `docs/galvo_code_map.md`，再进入具体目录修改。

## 目录结构

```
ICT/
├── docs/                      # 设计文档
│   ├── route-a-design.md     # 电子元器件路线设计
│   └── route-b-design.md     # 手部安全监控设计
│
├── training/                  # 历史训练脚本
│   └── route_a_electro/      # 旧版电子元器件训练链路
│
├── 2026_3_12/                 # 最新 yolo26 训练工作区
│   ├── train_yolo26n_gpu.py
│   └── yolo_dataset/
│
├── edge/                      # 上位机 / 端侧推理
│   ├── common/               # 通用模块（相机、振镜控制）
│   ├── route_a_app/          # 电子元器件分拣应用
│   ├── route_b_app/          # 手部安全监控应用
│   ├── unified_app/          # 统一检测应用
│   └── laser_galvo/          # 激光振镜系统 ⭐
│       ├── generate_calibration_target.py  # 标定板生成
│       ├── calibrate_galvo.py             # 自动标定
│       ├── galvo_controller.py            # 振镜控制器
│       ├── stm32_galvo_protocol.c         # STM32固件示例
│       ├── run.sh                         # 快速启动脚本
│       └── README.md                      # 详细文档
│
├── datasets/                  # 数据集
├── tools/                     # 辅助工具
│   ├── calibration_tool/     # 相机-振镜映射标定
│   └── capture_tool/         # 数据采集工具
│
├── mirror/                    # 单片机 / 板卡驱动
│   ├── stm32f401ccu6_dac8563/ # STM32F401 + DAC8563 正式固件
│   └── stm32_firmware_patch.c # 历史补丁参考（关键逻辑已合入 main.c）
│
├── calibration_data/          # 标定板与辅助图
│   ├── laser_calibration_board_A4.png     # 激光标定板
│   ├── laser_calibration_board_A4_coordinates.txt
│   └── checkerboard_calibration_A4.png    # 棋盘格标定板
│
└── models/                    # 转换后的部署模型文件
    ├── route_a_yolo26/       # 最新 yolo26 导出的 onnx / om（运行时生成）
    └── yolov8n_pose_aipp.om  # 人体姿态检测
```

## 快速开始

### 基础环境

**硬件要求:**
- Orange Pi AI Pro (Ascend 310B NPU)
- Intel RealSense D435 深度相机
- DAC8563 双通道振镜控制器（可选，用于激光标注）
- STM32微控制器（可选，用于激光控制）

**软件依赖:**
```bash
# Python环境
pip install opencv-python numpy pyrealsense2 mediapipe pyyaml pyserial

# Ascend工具链
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 场景1: 电子元器件检测分拣

```bash
# 1. 最新训练工作区（yolo26）
cd 2026_3_12
python train_yolo26n_gpu.py

# 2. 导出最新 yolo26 到 Ascend 310B OM
cd ..
bash tools/export_latest_yolo26_to_om.sh

# 3. 运行统一主线
bash start_unified.sh
```

### 场景2: 手部安全监控

**方案A: YOLOv8-Pose（NPU加速）**
```bash
cd edge/route_b_app
python hand_safety_monitor.py \
    --model ../../models/yolov8n_pose_aipp.om \
    --danger-distance 150 \
    --conf-thres 0.5
```

**方案B: MediaPipe Hands（CPU）**
```bash
cd edge/route_b_app
python hand_safety_monitor_mediapipe.py \
    --danger-distance 150
```

### 场景3: 统一检测 + 激光标注

**第一步：激光振镜标定**

```bash
cd edge/laser_galvo

# 1. 生成标定板（首次使用）
python3 generate_calibration_target.py

# 2. 连接硬件并执行自动标定（推荐）
./run_calibration.sh

# 或手动运行
python3 auto_calibrate.py --serial-port /dev/ttyUSB0

# 3. 测试标定精度
python3 calibrate_galvo.py \
    --test \
    --load ~/ICT/galvo_calibration.yaml
```

**第二步：运行完整系统**

```bash
cd edge/unified_app

# 不启用激光（仅检测）
python3 unified_monitor.py \
    --yolo-model ../../models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om \
    --data-yaml ../../config/yolo26_6cls.yaml \
    --danger-distance 300 \
    --conf-thres 0.55

# 启用激光标注
python3 unified_monitor.py \
    --yolo-model ../../models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om \
    --data-yaml ../../config/yolo26_6cls.yaml \
    --danger-distance 300 \
    --conf-thres 0.55 \
    --enable-laser \
    --laser-serial /dev/ttyUSB0 \
    --laser-calibration ../laser_galvo/galvo_calibration.yaml \
    --laser-min-score 0.7
```

**快捷启动:**
```bash
cd edge/laser_galvo
./run.sh  # 交互式菜单
```

## 激光振镜系统详解

### 系统架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  RealSense   │────▶│   YOLO物体   │────▶│   激光振镜   │
│   D435相机   │     │    检测      │     │   绘制框     │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
  [640x480图像]      [物体边界框]        [激光轨迹]
                       x1,y1,x2,y2      DAC值: 0-65535
```

### 坐标变换

通过单应性矩阵（Homography Matrix）实现像素坐标到振镜坐标的精确转换：

```python
[x_galvo]       [x_pixel]
[y_galvo] = H · [y_pixel]
[   1   ]       [   1   ]
```

标定精度: < 1mm（典型值）

### 串口协议

当前正式工作链路使用 **STM32 文本批量协议**：

- `0R,<x>,<y>,<w>,<h>;`：向槽位 `0` 写入矩形任务
- `1C,<x>,<y>,<r>;`：向槽位 `1` 写入圆形任务
- `U;`：提交任务缓冲区
- 可在一个串口包中连续发送多个任务，例如：
  `0C,1000,2000,500;1C,-2000,3000,800;U;`

说明：
- 现在是**索引在前、命令字母在后**，不是旧版 `R0...` / `C0...`
- `G/L` 属于旧固件校准命令，不再是当前正式协议的一部分

其中 `mirror/stm32f401ccu6_dac8563/Core/Src/main.c` 是当前正式实现；`edge/laser_galvo/stm32_galvo_protocol.c` 仅保留为早期二进制协议示例。

### 安全特性

- 手部距离 < 300mm 时自动禁用激光
- 激光打框冷却时间（默认2秒）
- 最多同时标注3个物体
- CRC16校验确保通信可靠性

## 性能指标

| 组件 | 指标 | 说明 |
|------|------|------|
| YOLO推理 | ~54ms (18.5 FPS) | Ascend 310B NPU |
| MediaPipe | ~103ms (9.7 FPS) | CPU运行 |
| 总系统帧率 | ~6 FPS | 包含所有组件 |
| 激光打框 | ~0.5s/框 | 20步/边 |
| 标定精度 | < 1mm | 典型值 |

## WebUI监控

所有应用都支持WebUI实时监控：

```bash
# 启动WebUI服务器（自动启动）
# 浏览器访问: http://ict.local:8001  (Route A)
#            http://ict.local:8002  (Unified)
#            http://ict.local:8003  (Route B)
```

## STM32固件集成

如需在STM32上使用激光振镜控制：

1. 正式固件目录位于 `mirror/stm32f401ccu6_dac8563/`
2. 串口协议入口位于 `Core/Src/main.c`
3. DAC 驱动位于 `HARDWARE/dac8563/`
4. 当前正式协议已切换为 `0C,...;1R,...;U;` 这种批量文本格式

详细分层说明见 `docs/galvo_code_map.md`

## 常见问题

### 1. 激光光斑检测失败

调整HSV阈值：
```python
# calibrate_galvo.py 中修改
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
```

### 2. 串口连接失败

检查权限：
```bash
sudo usermod -a -G dialout $USER
# 注销重新登录
```

### 3. NPU推理失败

确保环境变量：
```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 4. 标定精度不足

改进方法：
- 使用更平整的标定板（刚性材料）
- 改善照明条件
- 多次标定取平均

## 参考文档

- [Route A 设计文档](docs/route-a-design.md)
- [Route B 设计文档](docs/route-b-design.md)
- [激光振镜详细文档](edge/laser_galvo/README.md)
- [标定工具说明](tools/calibration_tool/README.md)

## 技术栈

- **推理框架**: Ascend CANN (ACL)
- **物体检测**: YOLOv8 (Ultralytics)
- **姿态检测**: YOLOv8-Pose / MediaPipe Hands
- **深度相机**: Intel RealSense D435
- **振镜控制**: DAC8563 (16-bit dual DAC)
- **通信协议**: UART with CRC16
- **坐标转换**: OpenCV Homography

## 开发建议

**环境配置:**
- Python 3.9+
- Ascend 310B 推理环境
- 使用 Docker 或 Conda 保持依赖一致

**开发流程:**
1. 在 `training/` 目录训练模型
2. 使用 ATC 转换为 OM 格式
3. 在 `edge/` 目录编写推理应用
4. 使用 `tools/` 进行标定和测试

## 许可证

本项目代码仅供学习和研究使用。

## 更新日志

### 2025-12-07
- 新增激光振镜物体标注系统
- 实现自动标定流程
- 添加统一检测应用（物体+手部+激光）
- 完善安全监控功能

### 2024-12-06
- 修复手部危险检测逻辑
- 优化WebUI服务器稳定性
- 更新文档结构

---

**版本:** 2.0
**最后更新:** 2025-12-07
