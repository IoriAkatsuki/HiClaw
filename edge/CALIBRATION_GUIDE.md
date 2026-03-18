# 激光振镜自动校准指南

## 概述

本文档说明如何对激光振镜系统进行自动校准，建立**像素坐标 → 振镜坐标**的精确映射关系。

## 振镜坐标系统

### 坐标定义
```
振镜坐标系（以零点为中心）:
     Y+
      ↑
      |
X- ←--+--→ X+
      |
      ↓
     Y-

- 中心点: (0, 0) = 振镜不偏转
- 数据类型: int16_t
- 范围: X: -32768 ~ +32767, Y: -32768 ~ +32767
- 单位: 振镜旋转单位（与DAC输出成正比）
```

### 命令示例
```
R0 5000,5000,2000,1000
└─┬─┴──┬───┴──┬───┘
  │    │      │
  │    │      └─ 矩形宽高: 2000 x 1000 单位
  │    └─ 中心坐标: (5000, 5000) - 右上方
  └─ 第0个图形，矩形

特殊位置:
- (0, 0)        : 中心
- (10000, 0)    : 右侧
- (-10000, 0)   : 左侧
- (0, 10000)    : 上方
- (0, -10000)   : 下方
```

## 校准原理

### 为什么不需要物距信息？

使用**单应性矩阵（Homography Matrix）**进行纯视觉标定：

```
像素坐标 [u, v, 1]ᵀ  →  [H]  →  振镜坐标 [x, y, 1]ᵀ

其中 H 是 3x3 矩阵，通过标定自动计算
```

**优点:**
- ✅ 无需知道相机内参
- ✅ 无需测量物理距离
- ✅ 自动补偿畸变
- ✅ 适应任何工作距离

### 校准流程

```
1. 生成标定网格（3x3共9个点）
   ┌─────┬─────┬─────┐
   │ •   │  •  │   • │  振镜位置
   ├─────┼─────┼─────┤  (-15000, -15000) 到
   │ •   │  •  │   • │  (+15000, +15000)
   ├─────┼─────┼─────┤
   │ •   │  •  │   • │
   └─────┴─────┴─────┘

2. 逐点打激光 → 相机捕获位置
   振镜: (-15000, -15000) → 像素: (120, 80)
   振镜: (0, 0)           → 像素: (320, 240)
   振镜: (+15000, +15000) → 像素: (520, 400)
   ...

3. 计算单应性矩阵 H
   使用 cv2.findHomography(像素点, 振镜点)

4. 保存到 YAML 文件
```

## 前置条件：更新STM32固件

### 需要添加的命令

当前STM32固件只支持 C/R/U 命令，需要添加 **G** 和 **L** 命令：

| 命令 | 格式 | 功能 | 用途 |
|------|------|------|------|
| **G** | `G<x>,<y>\n` | 立即移动到指定位置 | 校准时定位激光点 |
| **L** | `L0\n` / `L1\n` | 激光开关 | 校准时控制激光 |
| C | `C<i> <x>,<y>,<r>` | 绘制圆形 | 正常工作 |
| R | `R<i> <x>,<y>,<w>,<h>` | 绘制矩形 | 正常工作 |
| U | `U\n` | 更新执行 | 正常工作 |

### STM32固件修改

参考文件: `/home/oasis/Documents/ICT/mirror/stm32_firmware_patch.c`

在 `HAL_UARTEx_RxEventCallback()` 中添加：

```c
// G命令 - 立即移动
if (uart1_rx_buf[0] == 'G')
{
  int16_t x, y;
  sscanf((const char *)&uart1_rx_buf[1], "%hd,%hd", &x, &y);
  dac8563_output_int16(x, y);  // 立即移动振镜
}

// L命令 - 激光开关
if (uart1_rx_buf[0] == 'L')
{
  if (uart1_rx_buf[1] == '1')
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_SET);   // 开
  else
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_10, GPIO_PIN_RESET); // 关
}
```

## 校准步骤

### 1. 硬件准备

```bash
# 检查硬件连接
✓ STM32 已烧录更新后的固件（包含G/L命令）
✓ 振镜系统已连接到STM32
✓ 激光器已连接到PB10
✓ RealSense D435 或 USB摄像头已连接
✓ STM32通过USB串口连接到板卡
```

### 2. 连接测试

```bash
# 测试串口通信
cd /home/oasis/Documents/ICT/edge/laser_galvo

# 测试G命令（移动到中心）
echo "G0,0" > /dev/ttyUSB0

# 测试L命令（激光开关）
echo "L1" > /dev/ttyUSB0  # 打开
sleep 1
echo "L0" > /dev/ttyUSB0  # 关闭
```

### 3. 运行自动校准

```bash
cd /home/oasis/Documents/ICT/edge/laser_galvo

python3 calibrate_galvo.py \
    --serial-port /dev/ttyUSB0 \
    --baudrate 115200 \
    --output ~/ICT/galvo_calibration.yaml
```

**校准过程（自动）:**
1. 程序打开相机
2. 逐个标定点（3x3=9个点）:
   - 发送 `G<x>,<y>` 移动振镜
   - 发送 `L1` 打开激光
   - 相机检测激光光斑位置（红色）
   - 记录对应关系
3. 计算单应性矩阵
4. 保存到 `galvo_calibration.yaml`

**预期输出:**
```
============================================================
开始自动标定
============================================================
✓ 使用 RealSense D435
✓ 初始化 9 个振镜标定点
  坐标范围: (-15000, -15000) 到 (15000, 15000)
  中心点: (0, 0)

开始标定 9 个点...

点 1/9: 振镜位置 (-15000, -15000)
  ✓ 像素位置: (123.4, 87.6) ± (1.2, 0.8)

点 2/9: 振镜位置 (-15000, 0)
  ✓ 像素位置: (122.1, 239.3) ± (0.9, 1.1)
...

✓ 成功检测 9 个标定点

计算单应性矩阵...
✓ 单应性矩阵计算成功
  平均重投影误差: 234.5 振镜单位
  最大重投影误差: 456.7 振镜单位

✓ 标定数据已保存: galvo_calibration.yaml

============================================================
标定完成！
标定文件: galvo_calibration.yaml

运行测试:
  python3 calibrate_galvo.py --test --load galvo_calibration.yaml
============================================================
```

### 4. 测试校准精度

```bash
# 交互式测试
python3 calibrate_galvo.py --test --load ~/ICT/galvo_calibration.yaml
```

**测试方法:**
- 窗口显示相机画面
- 用鼠标点击任意位置
- 激光会移动到对应的物理位置
- 观察激光是否准确到达点击位置
- 按 `q` 退出

## 使用校准结果

### 在 unified_monitor.py 中使用

```bash
cd /home/oasis/Documents/ICT/edge/unified_app

python3 unified_monitor.py \
    --yolo-model ~/ICT/runs/detect/train_electro61/weights/yolov8_electro61_aipp.om \
    --data-yaml ~/ICT/config/electro61.yaml \
    --enable-laser \
    --laser-serial /dev/ttyUSB0 \
    --laser-calibration ~/ICT/galvo_calibration.yaml
```

### 在 galvo_controller.py 中使用

```python
from galvo_controller import LaserGalvoController

# 创建控制器（带标定文件）
galvo = LaserGalvoController(
    serial_port='/dev/ttyUSB0',
    baudrate=115200,
    calibration_file='~/ICT/galvo_calibration.yaml'
)

galvo.connect()

# 像素坐标自动转换为振镜坐标
box = [100, 100, 300, 300]  # 像素坐标
galvo.draw_box(box, pixel_coords=True)
galvo.update_tasks()
```

## 校准文件格式

`galvo_calibration.yaml`:
```yaml
homography_matrix:
  - [1.234, -0.056, -15234.5]
  - [0.023, 1.189, -14567.8]
  - [0.00001, -0.00002, 1.0]

galvo_points:  # 振镜标定点（9个）
  - [-15000, -15000]
  - [-15000, 0]
  - [-15000, 15000]
  ...

pixel_points:  # 对应的像素点
  - [123.4, 87.6]
  - [122.1, 239.3]
  ...

timestamp: '2025-12-07 18:30:45'
```

## 常见问题

### Q1: 激光光斑检测失败

**原因:**
- 激光颜色阈值不匹配
- 环境光太强
- 激光功率太弱

**解决:**
编辑 `calibrate_galvo.py` 第119-124行，调整HSV阈值：
```python
# 红色激光
lower_red1 = np.array([0, 100, 100])    # 调整这些值
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
```

### Q2: 重投影误差过大

**原因:**
- 相机镜头畸变严重
- 振镜安装不稳
- 工作平面不平整

**解决:**
1. 使用更多标定点（修改代码为5x5网格）
2. 固定相机和振镜位置
3. 使用平整的标定板

### Q3: STM32不响应G/L命令

**检查:**
```bash
# 监听STM32响应
screen /dev/ttyUSB0 115200

# 手动发送命令
G0,0
L1
L0
```

如果没有反应，说明固件未更新或命令格式错误。

## 高级选项

### 调整标定网格范围

编辑 `calibrate_galvo.py` 第45-46行：
```python
margin = 5000          # 边界裕度
coord_range = 15000    # 工作区域半径（可改为10000/20000等）
```

### 增加标定点数量

从3x3改为5x5：
```python
# 第49行
positions = [-coord_range, -coord_range/2, 0, coord_range/2, coord_range]
```

## 总结

校准完成后，系统能够：

✅ 将YOLO检测到的像素坐标自动转换为振镜坐标
✅ 精确地用激光标记检测到的物体
✅ 补偿相机畸变和安装误差
✅ 适应不同的工作距离和角度

**无需任何物理测量，完全自动化！**

---

**相关文件:**
- STM32固件补丁: `/mirror/stm32_firmware_patch.c`
- 校准程序: `/edge/laser_galvo/calibrate_galvo.py`
- 控制器: `/edge/laser_galvo/galvo_controller.py`
- 主程序: `/edge/unified_app/unified_monitor.py`
