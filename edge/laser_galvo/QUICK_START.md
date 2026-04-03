# 激光振镜校准 - 快速开始

## ✅ 你已准备好的：
- [x] A4校准板（9个圆形标记）
- [x] STM32已连接（ttyUSB/ttyACM）
- [x] 摄像头已架设

---

## 🚀 一键自动校准

```bash
cd /home/oasis/Documents/ICT/edge/laser_galvo

# 运行一键校准脚本
./run_calibration.sh
```

**校准过程（全自动）：**
1. ✅ 自动查找串口设备
2. ✅ 测试激光控制（L1/L0命令）
3. ✅ 预览相机画面（按's'开始）
4. ✅ 自动执行9点校准
5. ✅ 自动调整HSV参数
6. ✅ 保存到 `~/ICT/galvo_calibration.yaml`

---

## 📋 校准步骤详解

### 1. 准备环境

**摆放校准板：**
```
     相机（俯视）
        │
        ▼
   ┌──────────┐
   │ 1  2  3  │ ← A4纸平放
   │ 4  5  6  │
   │ 7  8  9  │
   └──────────┘
        ▲
     激光振镜
```

**检查清单：**
- [ ] 校准板平整放置
- [ ] 相机能看到所有9个标记
- [ ] 激光能照射到整个区域
- [ ] 光线均匀（避免强光直射）
- [ ] 相机和振镜固定（不能移动！）

### 2. 运行校准

```bash
# 方法1: 一键脚本（推荐）
./run_calibration.sh

# 方法2: 手动指定参数
python3 auto_calibrate.py
```

### 3. 校准过程

**自动进行：**
- 激光打到点1 (-15000, -15000) → 相机检测光斑位置
- 激光打到点2 (-15000, 0) → 相机检测
- ...
- 激光打到点9 (15000, 15000) → 相机检测
- 计算单应性矩阵
- 保存结果

**预期时间：** 2-3分钟

### 4. 验证结果

**交互测试：**
```bash
python3 calibrate_galvo.py \
    --test \
    --load ~/ICT/galvo_calibration.yaml
```

**操作：**
- 窗口显示相机画面
- 鼠标点击任意位置
- 激光移动到该位置
- 检查精度
- 按'q'退出

**精度标准：**
- ✅ 优秀: ±2mm
- ⚠️ 可接受: 2-5mm
- ❌ 需重新校准: >5mm

---

## 🎯 使用校准结果

### 在检测系统中使用

```bash
cd /home/oasis/Documents/ICT/edge/unified_app

# 带激光标记的物体检测
python3 unified_monitor.py \
    --yolo-model ~/ICT/models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om \
    --data-yaml ~/ICT/config/yolo26_6cls.yaml \
    --enable-laser \
    --laser-serial /dev/ttyUSB0 \
    --laser-calibration ~/ICT/edge/laser_galvo/galvo_calibration.yaml
```

### 独立使用

```python
from galvo_controller import LaserGalvoController

galvo = LaserGalvoController(
    serial_port='/dev/ttyUSB0',
    calibration_file='~/ICT/galvo_calibration.yaml'
)
galvo.connect()

# 绘制边界框（自动转换像素坐标）
boxes = [[100, 100, 200, 200]]
galvo.draw_boxes(boxes)

galvo.disconnect()
```

---

## ⚙️ 重要说明

### MCU命令协议

**你的STM32需要在每个命令后发送'U'更新：**

```
G1000,2000  ← 移动命令
U           ← 更新执行（必须！）

L1          ← 激光开
U           ← 更新执行（必须！）
```

**已自动处理：** `auto_calibrate.py` 和 `calibrate_galvo.py` 已自动在每个G/L命令后发送U。

### 自动调参功能

如果激光光斑检测失败，程序会自动尝试不同的HSV阈值：

```python
# 从宽松到严格
[(0, 50, 50), (10, 255, 255)]    # 第1次尝试
[(0, 80, 80), (10, 255, 255)]    # 第2次尝试
[(0, 100, 100), (10, 255, 255)]  # 第3次尝试
...
```

**支持最多3次重试，每次自动调整参数。**

---

## 🔧 故障排除

### 问题1: 串口连接失败

```bash
# 查看串口设备
ls -l /dev/ttyUSB* /dev/ttyACM*

# 检查权限
sudo chmod 666 /dev/ttyUSB0

# 手动测试
echo "L1" > /dev/ttyUSB0  # 激光应该亮
echo "U" > /dev/ttyUSB0
```

### 问题2: 激光不响应

**可能原因：** 固件未支持G/L命令

**解决：** 更新STM32固件，参考 `/home/oasis/Documents/ICT/mirror/stm32_firmware_patch.c`

### 问题3: 未检测到激光光斑

**解决方法（自动）：**
- 程序会自动尝试3次
- 每次使用不同的HSV阈值
- 如果都失败，手动调整环境光

**手动调整：**
编辑 `calibrate_galvo.py` 第119行：
```python
lower_red1 = np.array([0, 50, 50])   # 降低要求
upper_red1 = np.array([10, 255, 255])
```

### 问题4: 校准精度差

**原因：**
- 相机或振镜在校准中移动了
- 激光光斑检测不稳定

**解决：**
- 重新固定设备
- 增加/减少环境光
- 重新校准

---

## 📝 命令参考

### 串口测试命令

```bash
# 测试激光（手动）
echo -e "L1\nU" > /dev/ttyUSB0  # 开
sleep 1
echo -e "L0\nU" > /dev/ttyUSB0  # 关

# 测试移动
echo -e "G0,0\nU" > /dev/ttyUSB0     # 中心
echo -e "G5000,0\nU" > /dev/ttyUSB0  # 右侧
```

### 校准文件检查

```bash
# 查看校准文件
cat ~/ICT/galvo_calibration.yaml

# 验证校准质量
python3 << 'EOF'
import yaml, numpy as np
with open('/home/oasis/ICT/galvo_calibration.yaml') as f:
    data = yaml.safe_load(f)
print(f"标定点数: {len(data['pixel_points'])}")
print(f"标定时间: {data['timestamp']}")
EOF
```

---

## 📚 相关文档

- **完整校准指南**: `CALIBRATION_GUIDE.md`
- **系统集成文档**: `../LASER_INTEGRATION.md`
- **快速入门**: `../README_GALVO.md`
- **STM32固件补丁**: `/home/oasis/Documents/ICT/mirror/stm32_firmware_patch.c`

---

## 🎉 校准成功后

**校准文件位置：** `~/ICT/galvo_calibration.yaml`

**下一步：**
1. ✅ 测试精度
2. ✅ 在检测系统中使用
3. ✅ 享受自动激光标记！

**重要提醒：**
- ⚠️ 校准后**不要移动**相机或振镜！
- ⚠️ 如果移动了，需要**重新校准**
- ✅ 校准文件可以长期使用（硬件位置不变）

---

**准备好了吗？运行校准：**

```bash
cd /home/oasis/Documents/ICT/edge/laser_galvo
./run_calibration.sh
```
