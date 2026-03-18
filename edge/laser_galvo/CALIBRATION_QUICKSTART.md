# 激光振镜校准 - 快速开始（已有打印校准板）

## 前置条件检查

### ✅ 你已经准备好：
- [x] A4校准板（9个同心圆标记）
- [x] 摄像头（已连接 /dev/video0）

### ⚠️ 还需要：
- [ ] STM32振镜控制器通过USB连接
- [ ] STM32固件已更新（支持G/L命令）

---

## 第一步：连接STM32

### 1.1 硬件连接

```
电脑 USB ─► USB转串口 ─► STM32 USART1 (PA9/PA10)
                              │
                              ├─► DAC8563 (SPI) ─► 振镜驱动
                              └─► PB10 ─► 激光模块
```

### 1.2 连接后检查

```bash
# 插入USB后，执行：
ls -l /dev/ttyUSB* /dev/ttyACM*

# 应该看到类似：
# /dev/ttyUSB0 或 /dev/ttyACM0
```

**如果没有设备，参考**: `/tmp/stm32_connection_guide.md`

### 1.3 测试通信

```bash
# 方法1: 使用echo测试
echo "L1" > /dev/ttyUSB0  # 打开激光（应该看到激光亮）
sleep 1
echo "L0" > /dev/ttyUSB0  # 关闭激光

# 方法2: 使用screen监听
screen /dev/ttyUSB0 115200
# 输入: L1 回车（激光开）
# 输入: L0 回车（激光关）
# 退出: Ctrl-A, K
```

**⚠️ 重要**: 如果激光不响应，说明固件未更新G/L命令！

---

## 第二步：准备校准环境

### 2.1 摆放校准板

```
        相机 (俯视)
           │
           │
           ▼
    ┌──────────────┐
    │   [1] [2] [3]│  ← A4校准板
    │              │
    │   [4] [5] [6]│     平放在工作台上
    │              │
    │   [7] [8] [9]│
    └──────────────┘
           ▲
           │
        激光振镜
       (从侧面打光)
```

**注意事项:**
- ✅ 校准板平整放置
- ✅ 光线均匀（避免强光直射）
- ✅ 相机能清晰看到所有9个圆形标记
- ✅ 激光能够照射到校准板的所有区域
- ✅ 固定相机和振镜（校准期间不能移动！）

### 2.2 调整相机焦距

```bash
# 实时预览相机画面
python3 << 'EOF'
import cv2
cap = cv2.VideoCapture(0)
print("按 'q' 退出预览")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 调整大小便于查看
    frame_small = cv2.resize(frame, (640, 480))
    cv2.imshow('Camera Preview', frame_small)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
EOF
```

**检查:**
- ✅ 能看到所有9个圆形标记
- ✅ 标记清晰，边缘锐利
- ✅ 无模糊或反光

---

## 第三步：执行自动校准

### 3.1 运行校准程序

```bash
cd /home/oasis/Documents/ICT/edge/laser_galvo

# 确认串口设备名（通常是ttyUSB0）
SERIAL_PORT="/dev/ttyUSB0"  # 如果是ttyACM0，修改这里

# 运行校准
python3 calibrate_galvo.py \
    --serial-port $SERIAL_PORT \
    --baudrate 115200 \
    --output ~/ICT/galvo_calibration.yaml
```

### 3.2 校准过程（自动）

程序会自动：

```
1. 打开相机
2. 打开激光 (发送 L1)
3. 逐个标定点：
   点1: 发送 G-15000,-15000  (左上)
        等待振镜稳定
        相机检测激光光斑位置
        记录 振镜(-15000,-15000) ↔ 像素(x1, y1)

   点2: 发送 G-15000,0        (左中)
        ...

   点9: 发送 G15000,15000     (右下)
        ...

4. 关闭激光 (发送 L0)
5. 计算单应性矩阵
6. 保存到 galvo_calibration.yaml
```

**预期输出:**
```
============================================================
开始自动标定
============================================================
✓ 使用 USB 摄像头
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

✓ 标定数据已保存: /home/oasis/ICT/galvo_calibration.yaml

============================================================
标定完成！
============================================================
```

### 3.3 常见问题处理

**问题1: 未检测到激光光斑**

原因：激光颜色阈值不匹配

解决：按 `Ctrl+C` 停止，编辑 `calibrate_galvo.py` 第119-124行：

```python
# 调整HSV阈值（根据你的激光颜色）
lower_red1 = np.array([0, 50, 50])      # 降低饱和度/亮度要求
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])    # 调整这些值
upper_red2 = np.array([180, 255, 255])
```

**问题2: 激光不移动**

原因：固件不支持G命令

解决：更新STM32固件，参考 `/home/oasis/Documents/ICT/mirror/stm32_firmware_patch.c`

**问题3: 重投影误差过大**

原因：相机或振镜在校准过程中移动了

解决：重新固定设备，重新校准

---

## 第四步：测试校准

### 4.1 交互式测试

```bash
python3 calibrate_galvo.py \
    --test \
    --load ~/ICT/galvo_calibration.yaml
```

**操作:**
1. 窗口显示相机画面
2. 用鼠标**点击**任意位置
3. 激光会移动到对应的物理位置
4. 检查激光是否准确到达点击位置
5. 按 `q` 退出

**精度评估:**
- ✅ 优秀: 激光准确到达点击位置（±2mm）
- ⚠️ 可接受: 偏差 2-5mm
- ❌ 需重新校准: 偏差 >5mm

### 4.2 自动验证

```bash
python3 << 'EOF'
import yaml
import numpy as np

# 加载校准文件
with open('/home/oasis/ICT/galvo_calibration.yaml') as f:
    data = yaml.safe_load(f)

H = np.array(data['homography_matrix'])
galvo_pts = np.array(data['galvo_points'])
pixel_pts = np.array(data['pixel_points'])

# 重投影验证
pixel_pts_h = np.hstack([pixel_pts, np.ones((len(pixel_pts), 1))])
galvo_pred = (H @ pixel_pts_h.T).T
galvo_pred = galvo_pred[:, :2] / galvo_pred[:, 2:]

errors = np.linalg.norm(galvo_pred - galvo_pts, axis=1)
print(f"平均误差: {np.mean(errors):.1f} 振镜单位")
print(f"最大误差: {np.max(errors):.1f} 振镜单位")
print(f"最小误差: {np.min(errors):.1f} 振镜单位")

if np.mean(errors) < 500:
    print("\n✅ 校准质量: 优秀")
elif np.mean(errors) < 1000:
    print("\n⚠️ 校准质量: 可接受")
else:
    print("\n❌ 校准质量: 需重新校准")
EOF
```

---

## 第五步：使用校准结果

### 5.1 在检测系统中使用

```bash
cd /home/oasis/Documents/ICT/edge/unified_app

# 启动带激光标记的检测系统
python3 unified_monitor.py \
    --yolo-model ~/ICT/runs/detect/train_electro61/weights/yolov8_electro61_aipp.om \
    --data-yaml ~/ICT/config/electro61.yaml \
    --enable-laser \
    --laser-serial /dev/ttyUSB0 \
    --laser-calibration ~/ICT/galvo_calibration.yaml
```

### 5.2 单独使用振镜控制器

```python
from galvo_controller import LaserGalvoController

# 创建控制器（加载校准文件）
galvo = LaserGalvoController(
    serial_port='/dev/ttyUSB0',
    baudrate=115200,
    calibration_file='/home/oasis/ICT/galvo_calibration.yaml'
)

galvo.connect()

# 像素坐标自动转换
boxes = [
    [100, 100, 200, 200],  # 物体1
    [300, 150, 400, 250],  # 物体2
]

galvo.draw_boxes(boxes, image_width=640, image_height=480)

galvo.disconnect()
```

---

## 故障排除

### 串口问题
```bash
# 查看所有串口设备
ls -l /dev/tty{USB,ACM}*

# 检查权限
sudo chmod 666 /dev/ttyUSB0

# 测试连接
echo "L1" > /dev/ttyUSB0
```

### 相机问题
```bash
# 列出摄像头
v4l2-ctl --list-devices

# 测试捕获
python3 -c "import cv2; cap=cv2.VideoCapture(0); print('OK' if cap.read()[0] else 'FAIL')"
```

### 激光检测问题
- 调整HSV阈值（calibrate_galvo.py 第119行）
- 降低环境光
- 增加激光功率

---

## 下一步

✅ 校准完成后：
1. 测试精度
2. 保存校准文件（已自动保存）
3. 在检测系统中使用
4. 定期重新校准（如果移动了相机或振镜）

**重要提示**:
- 校准后**不要移动**相机或振镜！
- 如果移动了，需要**重新校准**
- 校准文件可以长期使用（只要硬件位置不变）
