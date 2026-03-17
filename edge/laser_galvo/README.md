# 激光振镜物体标注系统

基于物体检测的激光振镜自动标注系统，实现摄像头检测 → 坐标转换 → 激光打框的完整流程。

## 系统架构

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  RealSense   │────▶│   YOLO物体   │────▶│   激光振镜   │
│   D435相机   │     │    检测      │     │   绘制框     │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       │                     │                     │
       ▼                     ▼                     ▼
  [640x480图像]      [物体边界框]        [激光轨迹]
                       x1,y1,x2,y2      DAC值: 0-65535
```

## 文件说明

### 核心模块

1. **generate_calibration_target.py** - 标定板生成器
   - 生成A4打印标定板（3x3网格，9个标定点）
   - 输出: `laser_calibration_board_A4.png`

2. **calibrate_galvo.py** - 自动标定程序
   - 自动检测激光光斑位置
   - 建立像素-振镜坐标映射（单应性矩阵）
   - 输出: `galvo_calibration.yaml`

3. **galvo_controller.py** - 振镜控制器
   - 串口通信（CRC16校验）
   - 坐标转换（像素 → 振镜DAC值）
   - 激光轨迹绘制（box, line, move）

4. **stm32_galvo_protocol.c** - STM32固件示例
   - 串口协议解析
   - DAC8563控制接口
   - 激光开关控制

5. **../unified_app/unified_monitor.py** - 统一主入口（含激光参数）
   - YOLO物体检测
   - MediaPipe手部安全检测
   - 激光自动标注（避开危险情况）

## 快速开始

### 准备工作

**硬件需求:**
- Intel RealSense D435 深度相机
- DAC8563 双通道DAC振镜控制器
- STM32微控制器
- 激光器（带控制引脚）
- USB-串口转换器

**软件依赖:**
```bash
pip install opencv-python numpy pyrealsense2 pyserial pyyaml mediapipe
```

### 第一步：生成标定板

```bash
cd /home/oasis/Documents/ICT/edge/laser_galvo
python3 generate_calibration_target.py
```

**输出文件:**
- `/home/oasis/Documents/ICT/calibration_data/laser_calibration_board_A4.png`
- `/home/oasis/Documents/ICT/calibration_data/laser_calibration_board_A4_coordinates.txt`

**操作:**
1. 用A4纸打印 `laser_calibration_board_A4.png`
2. 平整贴在激光工作区域（建议用硬纸板做底）

### 第二步：STM32固件准备

**修改您的STM32项目:**

1. 将 `stm32_galvo_protocol.c` 添加到项目中

2. 在 `main.c` 中初始化:
```c
#include "galvo_protocol.h"

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_USART1_UART_Init();  // 确保串口已初始化
    MX_SPI1_Init();         // 如果DAC8563用SPI

    // 初始化振镜协议
    galvo_protocol_init();

    while (1) {
        // 串口处理在中断中完成
    }
}
```

3. 配置串口参数:
   - 波特率: 115200
   - 数据位: 8
   - 停止位: 1
   - 校验位: None

4. 根据实际硬件调整引脚定义:
```c
#define LASER_PORT      GPIOB
#define LASER_PIN       GPIO_PIN_0
```

5. 编译并烧录到STM32

### 第三步：执行自动标定

**连接硬件:**
```
STM32 ←[USB-串口]→ 上位机 (Ascend板)
  ↓
DAC8563 → 振镜 → 激光 → 标定板
  ↑                      ↓
  └───────[摄像头观测]────┘
```

**运行标定程序:**
```bash
cd /home/oasis/Documents/ICT/edge/laser_galvo
python3 calibrate_galvo.py \
    --serial-port /dev/ttyUSB0 \
    --baudrate 115200 \
    --output galvo_calibration.yaml
```

**标定流程:**
1. 程序打开激光
2. 依次移动到9个标定点
3. 摄像头检测每个激光光斑位置
4. 计算单应性矩阵（Homography Matrix）
5. 保存标定结果

**预期输出:**
```
============================================================
开始自动标定
============================================================
✓ 使用 RealSense D435
开始标定 9 个点...

点 1/9: 振镜位置 (8000, 8000)
  ✓ 像素位置: (102.3, 85.7) ± (1.2, 0.9)

点 2/9: 振镜位置 (32767, 8000)
  ✓ 像素位置: (320.1, 87.2) ± (0.8, 1.1)

...

✓ 成功检测 9 个标定点

计算单应性矩阵...
✓ 单应性矩阵计算成功
  平均重投影误差: 234.5 DAC单位
  最大重投影误差: 512.1 DAC单位
  平均误差: 0.47 mm
  最大误差: 1.02 mm

✓ 标定数据已保存: galvo_calibration.yaml
============================================================
标定完成！
标定文件: galvo_calibration.yaml
============================================================
```

### 第四步：测试标定精度

```bash
python3 calibrate_galvo.py \
    --serial-port /dev/ttyUSB0 \
    --test \
    --load galvo_calibration.yaml
```

**测试方法:**
1. 程序打开实时相机画面
2. 用鼠标点击图像中任意位置
3. 激光会移动到对应的实际位置
4. 检查激光是否准确指向点击位置
5. 按 'q' 退出

### 第五步：运行完整系统

```bash
cd /home/oasis/Documents/ICT/edge/unified_app

# 基本运行（不启用激光）
python3 unified_monitor.py \
    --yolo-model ../models/yolov8n_electro61.om \
    --data-yaml ../data/electro61.yaml \
    --danger-distance 300 \
    --conf-thres 0.55

# 启用激光标注
python3 unified_monitor.py \
    --yolo-model ../models/yolov8n_electro61.om \
    --data-yaml ../data/electro61.yaml \
    --danger-distance 300 \
    --conf-thres 0.55 \
    --enable-laser \
    --laser-serial /dev/ttyUSB0 \
    --laser-baudrate 115200 \
    --laser-calibration ../laser_galvo/galvo_calibration.yaml \
    --laser-min-score 0.7 \
    --laser-target-classes capacitor resistor IC
```

**参数说明:**
- `--enable-laser`: 启用激光打框功能
- `--laser-serial`: 串口设备路径
- `--laser-baudrate`: 串口波特率（默认 115200）
- `--laser-calibration`: 标定文件路径
- `--laser-min-score`: 只标注置信度高于此值的物体
- `--laser-target-classes`: 指定要标注的类别（可选，不指定则标注所有）

**安全特性:**
- 检测到手部距离 < 300mm 时，自动禁用激光
- 激光打框有冷却时间（默认2秒）
- 最多同时标注3个物体

## 串口协议说明

### 帧格式 (10字节)

```
[0xAA] [0x55] [CMD] [X_H] [X_L] [Y_H] [Y_L] [PARAM] [CRC_H] [CRC_L]
  ↑      ↑      ↑     ←─ X坐标 ─→ ←─ Y坐标 ─→   ↑      ←─ CRC16 ─→
 帧头1  帧头2  命令      16位        16位      参数       16位
```

### 命令定义

| 命令 | 值   | 功能           | X/Y参数      | PARAM |
|------|------|----------------|--------------|-------|
| MOVE | 0x01 | 移动到指定位置 | DAC坐标      | -     |
| LASER_ON | 0x02 | 打开激光   | -            | -     |
| LASER_OFF | 0x03 | 关闭激光  | -            | -     |
| DRAW_LINE | 0x04 | 绘制直线  | 终点坐标     | -     |
| DRAW_BOX | 0x05 | 绘制矩形   | 角点坐标     | -     |

### 示例命令

**移动到中心点 (32767, 32767):**
```
AA 55 01 7F FF 7F FF 00 [CRC_H] [CRC_L]
```

**打开激光:**
```
AA 55 02 00 00 00 00 00 [CRC_H] [CRC_L]
```

## 坐标系统

### 像素坐标系 (RealSense D435)
```
(0,0) ────────────▶ X (640)
  │
  │
  │
  ▼
  Y (480)
```

### 振镜坐标系 (DAC8563)
```
(0,0) ────────────▶ X (65535)
  │
  │
  │
  ▼
  Y (65535)
```

### 单应性变换

通过3x3矩阵H实现转换:
```
[x_galvo]       [x_pixel]
[y_galvo] = H · [y_pixel]
[   1   ]       [   1   ]
```

矩阵H通过标定获得，保存在 `galvo_calibration.yaml` 中。

## 故障排除

### 1. 激光光斑检测失败

**症状:** 标定时提示 "未检测到激光光斑"

**可能原因:**
- 激光太弱或相机过曝
- 激光颜色不是红色（需修改HSV阈值）
- 激光未对准摄像头视野

**解决方案:**
```python
# 在 calibrate_galvo.py 中调整HSV阈值
lower_red1 = np.array([0, 100, 100])    # 根据实际调整
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
```

### 2. 串口通信失败

**症状:** "串口连接失败" 或无响应

**检查清单:**
- [ ] 确认串口设备路径正确: `ls /dev/ttyUSB*`
- [ ] 检查波特率匹配 (115200)
- [ ] 确认用户有串口权限: `sudo usermod -a -G dialout $USER`
- [ ] STM32固件已烧录且正常运行
- [ ] 串口线连接正确 (TX-RX交叉)

**测试命令:**
```bash
# 测试串口回环
echo "test" > /dev/ttyUSB0

# 查看串口参数
stty -F /dev/ttyUSB0
```

### 3. 标定精度不足

**症状:** 重投影误差 > 2mm

**改进方法:**
1. 使用更平整的标定板（刚性材料）
2. 增加标定点数量（修改3x3为5x5网格）
3. 确保摄像头焦距正确
4. 改善照明条件
5. 多次标定取平均

### 4. 激光打框不准确

**症状:** 激光框与物体位置不匹配

**检查步骤:**
1. 重新运行标定测试: `python3 calibrate_galvo.py --test`
2. 确认摄像头和标定时使用同一位置
3. 检查标定板是否移动过
4. 验证YOLO检测框是否准确

### 5. 手部检测导致激光频繁停止

**症状:** 激光标注频繁被中断

**调整参数:**
```bash
# 减小危险距离
--danger-distance 200  # 改为20cm

# 或修改代码，只在手部非常接近时禁用
if depth_mm < 100:  # 仅在10cm以内禁用激光
    disable_laser = True
```

## 性能指标

### 系统性能

| 指标 | 数值 |
|------|------|
| YOLO推理时间 | ~54ms (18.5 FPS) |
| MediaPipe推理 | ~103ms (9.7 FPS) |
| 总帧率 | ~6 FPS |
| 激光打框时间 | ~0.5s/框 (20步/边) |
| 标定精度 | < 1mm (典型值) |

### 优化建议

1. **减少MediaPipe开销:**
   - 降低分辨率: `config.enable_stream(rs.stream.color, 320, 240, ...)`
   - 减少检测频率: 每3帧检测一次

2. **加快激光绘制:**
   - 提高串口波特率: `--laser-baudrate 921600`
   - 减少单次标注数量（默认最多3个）

> 说明：`draw_box(..., steps_per_edge=...)` 仅为兼容旧调用保留参数，
> 当前 STM32 文本协议不会按该参数进行插值绘制。

3. **多物体处理:**
   - 限制标注数量: `max_targets=3`
   - 优先高置信度: 按score降序排序

## 进阶功能

### 1. 激光图案绘制

```python
# 绘制圆形
def draw_circle(controller, center_x, center_y, radius, steps=36):
    controller.laser_on()
    for i in range(steps + 1):
        angle = 2 * np.pi * i / steps
        x = center_x + radius * np.cos(angle)
        y = center_y + radius * np.sin(angle)
        controller.move_to_pixel(int(x), int(y))
        time.sleep(0.01)
    controller.laser_off()
```

### 2. 物体追踪标注

```python
# 持续追踪单个物体
tracker = cv2.TrackerKCF_create()
tracker.init(frame, bbox)

while True:
    success, bbox = tracker.update(frame)
    if success:
        laser_controller.draw_box(bbox)
```

### 3. 多相机系统

```python
# 使用多个RealSense相机扩大视野
cameras = [
    RealSenseCamera(serial='123456789'),
    RealSenseCamera(serial='987654321')
]

# 每个相机独立标定
calibrations = [
    'galvo_calibration_cam1.yaml',
    'galvo_calibration_cam2.yaml'
]
```

## 引用和参考

**相关技术:**
- [YOLOv8 - Ultralytics](https://github.com/ultralytics/ultralytics)
- [MediaPipe Hands - Google](https://google.github.io/mediapipe/solutions/hands.html)
- [RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [DAC8563 Datasheet - Texas Instruments](https://www.ti.com/product/DAC8563)

**算法:**
- Homography Estimation: Zhang, Z. (2000). A flexible new technique for camera calibration.
- Object Detection: Redmon, J. et al. (2016). You Only Look Once: Unified, Real-Time Object Detection.

## 许可证

本项目代码仅供学习和研究使用。

## 技术支持

遇到问题请检查:
1. 本README的故障排除部分
2. 代码注释中的详细说明
3. 串口通信日志

---

**版本:** 1.0
**更新日期:** 2025-12-07
**作者:** Claude Code Assistant
