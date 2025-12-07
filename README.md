# Laser-AR Sorting on MindSpore & Orange Pi

基于 MindSpore 与香橙派 AI Pro 的激光可视分拣项目，集成物体检测、手部安全监控和激光振镜标注功能。

## 项目概述

本项目实现了三个主要应用场景：
- **Route A**: 电子元器件检测分拣（YOLO + 激光振镜）
- **Route B**: 手部安全监控（YOLOv8-Pose / MediaPipe）
- **Unified App**: 统一检测系统（物体检测 + 手部安全 + 激光标注）

## 目录结构

```
ICT/
├── docs/                      # 设计文档
│   ├── route-a-design.md     # 电子元器件路线设计
│   └── route-b-design.md     # 手部安全监控设计
│
├── training/                  # 训练端
│   └── route_a_electro/      # 电子元器件模型训练
│
├── edge/                      # 端侧推理
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
├── calibration_data/          # 标定数据
│   ├── laser_calibration_board_A4.png     # 激光标定板
│   ├── laser_calibration_board_A4_coordinates.txt
│   └── checkerboard_calibration_A4.png    # 棋盘格标定板
│
└── models/                    # 转换后的模型文件
    ├── yolov8n_electro61.om  # 电子元器件检测
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
# 1. 训练模型（可选，也可使用预训练模型）
cd training/route_a_electro
python train.py --data data.yaml --epochs 100

# 2. 转换模型
atc --model=yolov8n_electro61.onnx \
    --framework=5 \
    --output=yolov8n_electro61 \
    --soc_version=Ascend310B1

# 3. 运行检测
cd edge/route_a_app
python detect.py --model ../../models/yolov8n_electro61.om
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

# 1. 生成标定板
python3 generate_calibration_target.py
# 打印输出的 laser_calibration_board_A4.png

# 2. 连接硬件并执行自动标定
python3 calibrate_galvo.py \
    --serial-port /dev/ttyUSB0 \
    --baudrate 115200 \
    --output galvo_calibration.yaml

# 3. 测试标定精度
python3 calibrate_galvo.py \
    --test \
    --load galvo_calibration.yaml
```

**第二步：运行完整系统**

```bash
cd edge/unified_app

# 不启用激光（仅检测）
python3 unified_monitor.py \
    --yolo-model ../models/yolov8n_electro61.om \
    --data-yaml ../data/electro61.yaml \
    --danger-distance 300 \
    --conf-thres 0.55

# 启用激光标注
python3 unified_monitor_with_laser.py \
    --yolo-model ../models/yolov8n_electro61.om \
    --data-yaml ../data/electro61.yaml \
    --danger-distance 300 \
    --conf-thres 0.55 \
    --enable-laser \
    --laser-serial /dev/ttyUSB0 \
    --laser-calibration ../laser_galvo/galvo_calibration.yaml \
    --laser-min-score 0.7 \
    --laser-target-classes capacitor resistor IC
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

10字节帧格式：
```
[0xAA][0x55][CMD][X_H][X_L][Y_H][Y_L][PARAM][CRC_H][CRC_L]
```

命令定义：
- `0x01` MOVE: 移动到指定位置
- `0x02` LASER_ON: 打开激光
- `0x03` LASER_OFF: 关闭激光
- `0x04` DRAW_LINE: 绘制直线
- `0x05` DRAW_BOX: 绘制矩形

详细协议说明见 `edge/laser_galvo/README.md`

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

1. 将 `edge/laser_galvo/stm32_galvo_protocol.c` 添加到项目
2. 在 `main.c` 中调用 `galvo_protocol_init()`
3. 配置串口（115200, 8N1）
4. 调整引脚定义匹配实际硬件

详细说明见 `edge/laser_galvo/stm32_galvo_protocol.c` 注释

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
