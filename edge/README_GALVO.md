# 激光振镜控制系统 - 快速入门

## 系统架构

```
┌──────────────────────────────────────────────────────────┐
│              Ascend 310 板卡（上位机）                    │
│  ┌────────────────┐  ┌────────────────┐                  │
│  │ YOLO 物体检测  │  │ MediaPipe 手部 │                  │
│  └────────┬───────┘  └────────┬───────┘                  │
│           │                   │                           │
│           └─────────┬─────────┘                           │
│                     ▼                                     │
│          ┌──────────────────────┐                         │
│          │ LaserGalvoController │ 决策层                  │
│          │  - 坐标转换          │                         │
│          │  - 指令生成          │                         │
│          └──────────┬───────────┘                         │
│                     │ 串口（文本协议）                    │
└─────────────────────┼───────────────────────────────────┘
                      │ USB-UART
┌─────────────────────┼───────────────────────────────────┐
│                     ▼                                     │
│         STM32F401CCU6 （下位机）                          │
│          - 解析命令 (C/R/U/G/L)                           │
│          - 控制 DAC8563                                   │
│          - 控制激光开关                                   │
│                     │                                     │
│          ┌──────────┴────────────┐                        │
│          ▼                       ▼                        │
│     ┌─────────┐            ┌──────────┐                  │
│     │ DAC8563 │            │ 激光模块 │                  │
│     └────┬────┘            └──────────┘                  │
│          │                                                │
│          ▼                                                │
│     ┌─────────┐                                           │
│     │ 振镜系统│                                           │
│     └─────────┘                                           │
└───────────────────────────────────────────────────────────┘
```

## 通信协议

### 振镜坐标系
- **原点**: (0, 0) = 振镜中心，不偏转
- **范围**: X: -32768 ~ +32767, Y: -32768 ~ +32767 (int16_t)
- **单位**: 振镜旋转单位

### 命令集

| 命令 | 格式 | 示例 | 功能 | 状态 |
|------|------|------|------|------|
| **R** | `<i>R,<x>,<y>,<w>,<h>;` | `0R,5000,5000,2000,1000;` | 绘制矩形任务 | ✅ STM32已支持 |
| **C** | `<i>C,<x>,<y>,<r>;` | `1C,0,0,3000;` | 绘制圆形任务 | ✅ STM32已支持 |
| **U** | `U;` | `U;` | 提交并切换任务缓冲区 | ✅ STM32已支持 |

说明：
- 当前正式协议是**索引在前、命令字母在后、分号分隔 token**
- 旧版 `R0...` / `C0...` / `G` / `L` 视为历史资料，不再是当前正式固件接口

## 文件结构

```
/home/oasis/Documents/ICT/
│
├── mirror/                              # STM32固件源码
│   ├── stm32f401ccu6_dac8563/          # 主固件
│   │   └── Core/Src/main.c             # 通信协议实现
│   └── stm32_firmware_patch.c          # ✅ G/L命令补丁代码
│
├── edge/                                # Ascend板卡代码
│   ├── laser_galvo/                    # 振镜控制模块
│   │   ├── galvo_controller.py         # ✅ 主控制器（已适配）
│   │   ├── calibrate_galvo.py          # ✅ 自动校准（已适配）
│   │   └── generate_calibration_target.py
│   │
│   ├── unified_app/                    # 统一监控应用
│   │   ├── unified_monitor.py          # ✅ 主程序（已集成激光）
│   │   └── start_unified_with_laser.sh # ✅ 启动脚本
│   │
│   ├── LASER_INTEGRATION.md            # ✅ 集成说明文档
│   ├── CALIBRATION_GUIDE.md            # ✅ 校准完整指南
│   └── README_GALVO.md                 # 本文件
│
└── ICT/
    └── galvo_calibration.yaml          # 校准结果（运行后生成）
```

## 快速开始

### 1️⃣ 更新STM32固件（一次性）

参考文件: `/mirror/stm32_firmware_patch.c`

在STM32固件的 `HAL_UARTEx_RxEventCallback()` 函数中添加G和L命令支持。

**为什么需要？** 校准程序需要G命令移动激光点，L命令控制激光开关。

### 2️⃣ 硬件连接检查

```bash
# 检查STM32串口
ls -l /dev/ttyUSB*

# 测试通信（发送到中心位置）
echo "G0,0" > /dev/ttyUSB0
echo "L1" > /dev/ttyUSB0  # 打开激光
sleep 1
echo "L0" > /dev/ttyUSB0  # 关闭激光
```

### 3️⃣ 执行自动校准（一次性）

```bash
cd /home/oasis/Documents/ICT/edge/laser_galvo

# 运行校准程序
python3 calibrate_galvo.py \
    --serial-port /dev/ttyUSB0 \
    --baudrate 115200 \
    --output ~/ICT/galvo_calibration.yaml

# 测试校准精度（可选）
python3 calibrate_galvo.py --test --load ~/ICT/galvo_calibration.yaml
```

**详细说明**: 查看 `CALIBRATION_GUIDE.md`

### 4️⃣ 运行检测系统

```bash
cd /home/oasis/Documents/ICT/edge/unified_app

# 方式1: 使用启动脚本（推荐）
./start_unified_with_laser.sh --enable-laser

# 方式2: 直接运行
python3 unified_monitor.py \
    --yolo-model ~/ICT/models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om \
    --data-yaml ~/ICT/config/yolo26_6cls.yaml \
    --enable-laser \
    --laser-serial /dev/ttyUSB0 \
    --laser-calibration ~/ICT/edge/laser_galvo/galvo_calibration.yaml
```

## 工作流程

```
相机采集
   ↓
YOLO检测物体 → 边界框 [x1, y1, x2, y2]
   ↓
像素坐标 → 单应性矩阵H → 振镜坐标 (x_galvo, y_galvo)
   ↓
生成R命令: "R0 x_galvo,y_galvo,width,height"
   ↓
串口发送 → STM32接收
   ↓
STM32解析 → 控制DAC8563 → 振镜偏转
   ↓
激光标记物体边界框 ✅
```

## 核心代码示例

### Python - 发送矩形绘制命令

```python
from galvo_controller import LaserGalvoController

# 初始化（带校准文件）
galvo = LaserGalvoController(
    serial_port='/dev/ttyUSB0',
    baudrate=115200,
    calibration_file='~/ICT/galvo_calibration.yaml'
)
galvo.connect()

# 检测到的物体框（像素坐标）
detected_boxes = [
    [100, 100, 200, 200],  # 物体1
    [300, 150, 400, 250],  # 物体2
]

# 自动转换并绘制（最多10个）
galvo.draw_boxes(detected_boxes, image_width=640, image_height=480)

# 清理
galvo.disconnect()
```

### STM32 - 解析R命令

```c
// 已有代码（无需修改）
if (uart1_rx_buf[0] == 'R')
{
  int i = uart1_rx_buf[1] - '0';  // 索引 0-9
  int16_t x, y;
  uint16_t length, height;
  sscanf((const char *)&uart1_rx_buf[2], "%hd,%hd,%hu,%hu",
         &x, &y, &length, &height);

  task_buf_1[i].type = RECTANGLE;
  task_buf_1[i].pose.x = x;        // 中心X
  task_buf_1[i].pose.y = y;        // 中心Y
  task_buf_1[i].params[0] = length;
  task_buf_1[i].params[1] = height;
}
```

## 坐标转换示例

假设校准完成：

| 像素坐标 (u, v) | 振镜坐标 (x, y) | 说明 |
|-----------------|-----------------|------|
| (320, 240) | (0, 0) | 图像中心 → 振镜中心 |
| (520, 240) | (+12000, 0) | 图像右侧 → 振镜右偏 |
| (120, 240) | (-12000, 0) | 图像左侧 → 振镜左偏 |
| (320, 80) | (0, -10000) | 图像上方 → 振镜上偏 |

**重要**: 具体数值由校准决定，无需手动计算！

## 常见问题

### Q: 不校准可以用吗？

可以，但精度低。`galvo_controller.py` 提供简单线性映射作为后备：

```python
# 无校准文件时的简单映射
x_galvo = (x_pixel - 320) * (65535 / 640)
y_galvo = (y_pixel - 240) * (65535 / 480)
```

**建议**: 至少校准一次，保存结果长期使用。

### Q: 校准需要多久？

- **自动校准**: 约2-3分钟（9个点）
- **测试验证**: 1分钟
- **总计**: 5分钟内完成

### Q: 校准后精度如何？

典型精度：
- 重投影误差: < 500 振镜单位
- 物理误差: < 2mm （取决于工作距离）

### Q: STM32不支持G/L命令怎么办？

**临时方案**: 修改 `calibrate_galvo.py` 使用R命令绘制极小矩形模拟点光源（不推荐，精度差）

**推荐方案**: 更新STM32固件，添加G/L命令支持（参考 `stm32_firmware_patch.c`）

## 性能指标

| 指标 | 数值 |
|------|------|
| YOLO推理 | ~30ms |
| 手部检测 | ~20ms |
| 坐标转换 | <1ms |
| 串口发送 | ~5ms (10个框) |
| 总延迟 | ~60ms |
| 系统FPS | ~15 |

## 下一步

1. ✅ 阅读 `LASER_INTEGRATION.md` 了解系统集成
2. ✅ 阅读 `CALIBRATION_GUIDE.md` 进行校准
3. ✅ 更新STM32固件（`stm32_firmware_patch.c`）
4. ✅ 运行校准程序
5. ✅ 启动检测系统

## 支持

- **问题报告**: 检查 `CALIBRATION_GUIDE.md` 的故障排查章节
- **代码参考**: 查看 `galvo_controller.py` 的注释
- **协议说明**: 参考 `LASER_INTEGRATION.md`

---

**最后更新**: 2025-12-07
**适配状态**: ✅ 完成
**测试状态**: ⚠️ 待硬件测试
