# 激光振镜与YOLO集成指南

## 映射关系说明

根据你的校准结果，系统使用以下映射关系：

```
X方向：1像素 = 102.4 个振镜单位
Y方向：1像素 = 136.5 个振镜单位
图像中心(320, 240) → 振镜坐标(0, 0)
```

### 坐标转换公式

```python
# 像素坐标 → 振镜坐标
galvo_x = (pixel_x - 320) * 102.4
galvo_y = (pixel_y - 240) * 136.5

# 宽度/高度转换
galvo_width = pixel_width * 102.4
galvo_height = pixel_height * 136.5
```

## 使用方法

### 1. 在 unified_monitor.py 中集成

代码已经集成完毕，使用方法：

```bash
cd /home/HwHiAiUser/ICT
python3 unified_monitor.py \
    --yolo-model models/yolov11_electro61_aipp.om \
    --data-yaml config/electro61.yaml \
    --enable-laser \
    --laser-serial /dev/ttyUSB0
```

### 2. 工作流程

```
YOLO检测 → 获取box坐标 → 转换为振镜坐标 → 发送串口命令 → 激光绘制
```

### 3. 串口命令格式

根据mirror中的STM32固件，命令格式为：

```
R<index> <x>,<y>,<width>,<height>\n
U\n
```

**示例**：
```python
# YOLO检测到物体，box = [100, 80, 200, 180]
# 计算：
#   中心点：(150, 130)
#   尺寸：100x100
# 转换为振镜坐标：
#   中心点：(150-320)*102.4, (130-240)*136.5 = (-17408, -15015)
#   尺寸：100*102.4, 100*136.5 = (10240, 13650)

# 发送命令：
"R0 -17408,-15015,10240,13650\n"
"U\n"
```

## 代码示例

### 简单示例（单个box）

```python
from edge.laser_galvo.galvo_controller import LaserGalvoController

# 创建控制器
galvo = LaserGalvoController('/dev/ttyUSB0', 115200)
galvo.connect()

# YOLO检测到的box（像素坐标）
box = [100, 80, 200, 180]  # [x1, y1, x2, y2]

# 绘制（自动转换坐标）
galvo.draw_box(box, pixel_coords=True, image_width=640, image_height=480)
galvo.update_tasks()  # 发送U命令更新显示
```

### 批量绘制多个box

```python
# YOLO检测到多个物体
boxes = [
    [100, 80, 200, 180],
    [300, 150, 400, 250],
    [450, 300, 550, 400],
]

# 批量绘制（最多10个）
galvo.draw_boxes(boxes, image_width=640, image_height=480)
# draw_boxes会自动发送U命令
```

### 在unified_monitor.py中的实际使用

```python
# 检测到物体后
if len(objects) > 0 and galvo:
    # 提取box列表
    boxes = [obj['box'] for obj in objects]  # box格式：[x1, y1, x2, y2]

    # 批量绘制
    galvo.draw_boxes(boxes, image_width=frame_width, image_height=frame_height)
```

## 坐标验证

测试结果显示映射正确：

| 像素坐标 | 振镜坐标 | 说明 |
|---------|---------|------|
| (320, 240) | (0, 0) | 画面中心 |
| (150, 150) | (-17408, -12285) | 左上角 |
| (490, 330) | (17408, 12285) | 右下角 |

## 坐标范围

- **像素范围**：640×480
- **振镜范围**：-32768 到 32767（int16_t）
- **有效工作区**：
  - X: ±32768 / 102.4 ≈ ±320 像素
  - Y: ±32767 / 136.5 ≈ ±240 像素
  - 即整个640×480画面都在工作范围内 ✓

## 性能优化建议

1. **批量发送**：多个box一次性发送，最后统一U更新
2. **限制数量**：STM32最多支持10个任务，超过的会被忽略
3. **串口权限**：首次使用需要：
   ```bash
   sudo chmod 666 /dev/ttyUSB0
   ```

## 故障排查

### 问题1：串口权限错误
```bash
sudo chmod 666 /dev/ttyUSB0
```

### 问题2：激光不显示
- 检查L1命令是否发送（激光开启）
- 检查U命令是否发送（更新显示）
- 检查坐标是否在范围内

### 问题3：位置偏移
- 确认映射系数（102.4, 136.5）
- 确认图像分辨率（640×480）
- 检查坐标计算是否减去中心点

## 测试脚本

测试映射是否正确：
```bash
cd /home/HwHiAiUser/ICT
python3 test_galvo_mapping.py
```

该脚本会在5个不同位置绘制矩形框，验证映射关系。

## 相关文件

- `edge/laser_galvo/galvo_controller.py` - 控制器主文件
- `edge/unified_app/unified_monitor.py` - 集成YOLO的主程序
- `mirror/stm32f401ccu6_dac8563/` - STM32固件（参考）
