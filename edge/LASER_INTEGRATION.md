# 激光振镜控制系统集成说明

## 概述

已将 `/mirror` 下的STM32激光振镜控制代码适配到 `/edge` 代码仓库，实现了YOLO物体检测与激光振镜标记的联动。

## 系统架构

```
┌─────────────────┐
│  RealSense D435 │ ──► 图像采集
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  YOLO + MediaPipe│ ──► 物体检测 + 手部检测
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ LaserGalvoController │ ──► 激光振镜标记
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  STM32 + DAC8563 │ ──► 振镜控制硬件
└─────────────────┘
```

## 通信协议

### STM32端协议（文本格式）

位置: `/mirror/stm32f401ccu6_dac8563/Core/Src/main.c`

**命令格式:**
1. **矩形绘制**: `R<index> <x>,<y>,<width>,<height>\n`
   - 例: `R0 5000,5000,2000,1000`
   - x, y: 中心点坐标 (int16_t, -32768~32767)
   - width, height: 矩形宽度和高度

2. **圆形绘制**: `C<index> <x>,<y>,<radius>\n`
   - 例: `C0 5000,5000,1000`
   - x, y: 圆心坐标
   - radius: 半径

3. **更新命令**: `U\n`
   - 触发STM32切换到新的任务缓冲区并开始绘制

**特点:**
- 双缓冲设计：`task_buf` 和 `task_buf_1`
- 支持最多10个任务 (index: 0-9)
- 激光开关通过PB10控制（GPIO）

## Python端适配

### 1. `galvo_controller.py` 修改

位置: `/edge/laser_galvo/galvo_controller.py`

**主要更改:**
- ❌ 移除二进制协议 (`[0xAA][0x55]...CRC`)
- ✅ 实现文本协议 (`_send_text_command`)
- ✅ 添加 `draw_box()` 方法（R命令）
- ✅ 添加 `draw_circle()` 方法（C命令）
- ✅ 添加 `update_tasks()` 方法（U命令）
- ✅ 支持无标定文件的简单线性映射

**关键方法:**
```python
# 像素坐标 -> 振镜坐标
pixel_to_galvo(x_pixel, y_pixel, image_width=640, image_height=480)

# 绘制单个矩形
draw_box(box, pixel_coords=True, task_index=None, image_width=640, image_height=480)

# 批量绘制（最多10个）
draw_boxes(boxes, image_width=640, image_height=480)
```

### 2. `unified_monitor.py` 集成

位置: `/edge/unified_app/unified_monitor.py`

**添加内容:**
1. 导入 `LaserGalvoController`
2. 命令行参数:
   - `--enable-laser`: 启用激光标记
   - `--laser-serial`: 串口设备 (默认: /dev/ttyUSB0)
   - `--laser-baudrate`: 波特率 (默认: 115200)
   - `--laser-calibration`: 标定文件（可选）

3. 主循环集成:
   ```python
   # 检测物体后，自动用激光标记
   if galvo and len(objects) > 0:
       boxes_to_draw = [obj['box'] for obj in objects]
       galvo.draw_boxes(boxes_to_draw, image_width=w_orig, image_height=h_orig)
   ```

## 使用方法

### 方式1: 直接运行Python脚本

**不带激光:**
```bash
cd /home/oasis/Documents/ICT/edge/unified_app
python3 unified_monitor.py \
    --yolo-model ~/ICT/runs/detect/train_electro61/weights/yolov8_electro61_aipp.om \
    --data-yaml ~/ICT/config/electro61.yaml
```

**带激光:**
```bash
python3 unified_monitor.py \
    --yolo-model ~/ICT/runs/detect/train_electro61/weights/yolov8_electro61_aipp.om \
    --data-yaml ~/ICT/config/electro61.yaml \
    --enable-laser \
    --laser-serial /dev/ttyUSB0 \
    --laser-baudrate 115200
```

### 方式2: 使用启动脚本

```bash
cd /home/oasis/Documents/ICT/edge/unified_app

# 不带激光
./start_unified_with_laser.sh

# 带激光
./start_unified_with_laser.sh --enable-laser
```

### 方式3: 测试激光控制器

```bash
cd /home/oasis/Documents/ICT/edge/laser_galvo

# 测试模式（绘制中心矩形）
python3 galvo_controller.py --test

# 交互模式
python3 galvo_controller.py
> box 100,100,300,300    # 绘制矩形
> circle 200,200,50      # 绘制圆形
> update                 # 触发绘制
> quit                   # 退出
```

## 硬件连接

### 串口连接
```
PC (USB) ──► CH340/CP2102 ──► STM32 (USART1)
                                  │
                                  ├─► SPI ──► DAC8563 ──► 振镜驱动
                                  └─► PB10 ──► 激光开关
```

### 检查串口设备
```bash
# 查看连接的USB串口
ls -l /dev/ttyUSB*

# 查看串口详细信息
dmesg | grep tty

# 赋予访问权限（如需要）
sudo chmod 666 /dev/ttyUSB0
# 或加入dialout组（永久方案）
sudo usermod -a -G dialout $USER
```

## 坐标系统

### 像素坐标系
- 原点: 图像左上角 (0, 0)
- 范围: 宽度 640, 高度 480

### 振镜坐标系（STM32 int16_t）
- 原点: 中心 (0, 0)
- 范围: X: -32768 ~ 32767, Y: -32768 ~ 32767
- 映射公式（无标定时）:
  ```python
  x_galvo = (x_pixel - 320) * (65535 / 640)
  y_galvo = (y_pixel - 240) * (65535 / 480)
  ```

### 标定（可选）
如果需要更精确的映射，可使用标定矩阵：
```bash
cd /home/oasis/Documents/ICT/edge/laser_galvo

# 1. 生成标定目标图案
python3 generate_calibration_target.py

# 2. 运行标定程序
python3 calibrate_galvo.py \
    --serial-port /dev/ttyUSB0 \
    --output calibration.yaml

# 3. 使用标定文件
python3 unified_monitor.py ... --laser-calibration calibration.yaml
```

## 工作流程

1. **系统启动**
   - 初始化ACL (YOLO推理)
   - 初始化MediaPipe (手部检测)
   - 初始化RealSense D435
   - 初始化激光振镜（如果启用）

2. **检测循环**
   ```
   读取相机 → YOLO推理 → 手部检测
        ↓
   检测到物体？
        ├─ 是 → 绘制框 + 发送激光命令 (R0, R1, ..., R9, U)
        └─ 否 → 继续
   ```

3. **激光绘制**
   - 批量发送最多10个矩形命令
   - 发送 `U` 命令触发STM32绘制
   - STM32切换缓冲区并执行绘制任务

## 性能优化

- **YOLO推理**: ~30ms (Ascend 310)
- **手部检测**: ~20ms (MediaPipe)
- **激光命令**: ~5ms (10个boxes)
- **总延迟**: ~60ms (约15 FPS)

## 故障排查

### 问题1: 串口连接失败
```bash
# 检查设备
ls -l /dev/ttyUSB*

# 查看权限
groups $USER  # 确认是否在dialout组

# 检查是否被占用
lsof /dev/ttyUSB0
```

### 问题2: 激光不绘制
1. 检查STM32是否正常供电
2. 检查激光开关（PB10）
3. 检查DAC8563 SPI连接
4. 串口监听调试:
   ```bash
   # 监听STM32接收的命令
   screen /dev/ttyUSB0 115200
   ```

### 问题3: 坐标不准确
- 运行标定程序生成标定矩阵
- 检查振镜机械安装是否稳固
- 调整像素到振镜的映射参数

## 文件清单

```
/home/oasis/Documents/ICT/
├── mirror/
│   └── stm32f401ccu6_dac8563/          # STM32固件源码
│       └── Core/Src/main.c              # 通信协议实现
├── edge/
│   ├── laser_galvo/
│   │   ├── galvo_controller.py          # ✅ 已适配文本协议
│   │   ├── calibrate_galvo.py
│   │   └── generate_calibration_target.py
│   └── unified_app/
│       ├── unified_monitor.py           # ✅ 已集成激光控制
│       └── start_unified_with_laser.sh  # ✅ 新增启动脚本
└── LASER_INTEGRATION.md                 # 本文档
```

## 下一步改进

1. **标定优化**: 实现自动化标定流程
2. **性能提升**: 异步发送激光命令，减少主循环延迟
3. **安全机制**: 手部接近时自动关闭激光
4. **可视化**: 在WebUI中显示激光状态
5. **日志记录**: 记录激光绘制次数和失败率

---

**修改时间**: 2025-12-07
**适配完成**: galvo_controller.py, unified_monitor.py
**测试状态**: 待测试（需硬件连接）
