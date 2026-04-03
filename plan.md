# 基于 MindSpore + 香橙派 AI Pro 的电子元器件分拣方案设计

> 说明：本文件用于本地仓库初始化项目框架，专注于“路线 A：电子元器件智能识别与激光可视分拣”。

---

## 0. 项目总体说明

- **项目名称（暂定）**：Laser-AR Sorting on MindSpore & Orange Pi
- **硬件平台**：
  - 香橙派 AI Pro 20T（Ascend 310B NPU）
  - Intel RealSense D435i（RGB + Depth）
  - 激光振镜 + 控制卡（EZCAD 兼容或国产控制板）
  - 喂料平台：托盘或低速输送带
  - USB 麦克风 + 扬声器（预留语音 agent）
- **软件栈**：
  - OS：Ubuntu / openEuler（根据板卡官方镜像选择）
  - 训练端：MindSpore 2.6+ / MindYOLO / Python 3
  - 推理端：MindSpore Lite / Ascend CANN 推理
  - 语言：Python 为主（C++ 可选，用于高性能模块）
  - 工具：Docker、Git、Gitee/GitHub

- **总体目标**：
  - 在香橙派 AI Pro 上通过 MindSpore 部署目标检测/异常检测模型，实现对多种工业电子元器件的实时识别；
  - 通过激光振镜将检测结果以图形形式投射到实物上，实现“激光 AR”式的可视分拣；
  - 预留本地语音 agent 能力，通过语音命令触发检测和打标。

- **仓库结构建议（示例）**：

```text
.
├── docs/                      # 设计文档、说明书
│   └── route-a-design.md
├── training/                  # 训练端代码（云端/PC）
│   └── route_a_electro/       # 电子元器件
├── edge/                      # 端侧推理与控制
│   ├── common/                # 公共模块（推理封装、标定、激光控制等）
│   └── route_a_app/           # 电子元器件端侧应用
├── datasets/                  # 数据集下载脚本 / 说明
├── tools/                     # 数据采集、标注、标定小工具
└── README.md
```

---

## 1. 公共基础模块设计

### 1.1 硬件抽象与依赖

- `edge/common/hw_config.yaml`
  - 相机内参、外参（D435i）
  - 振镜工作范围与分辨率
  - 板卡性能参数（NPU/CPU 资源预算）

- `edge/common/realsense_capture/`
  - 封装 D435i SDK，输出同步的 RGB + Depth 帧
  - 支持触发模式 / 定时采样

- `edge/common/ms_infer/`
  - MindSpore Lite 推理包装：模型加载、预处理、推理、后处理接口
  - 封装 YOLO 类模型输出：bbox、score、class_id

- `edge/common/calibration/`
  - 标定流程工具：
    - 相机像素坐标 → 工作平面物理坐标
    - 物理坐标 → 振镜坐标
  - 支持：
    - 2D Homography（单平面假设）
    - 可选 3D 外参（利用 depth 做高度补偿）

- `edge/common/laser_controller/`
  - 底层通信：串口或 TCP 与振镜控制卡交互
  - 上层接口：
    - `draw_rectangle(x, y, w, h)`
    - `draw_circle(x, y, r)`
    - `draw_polygon(points)`
    - `clear()` / `flash()` 等

- `edge/common/ui_web/`
  - 简易 Web 或本地 GUI：实时显示相机画面及检测框
  - 显示各类目标数量统计

- `tools/calibration_tool/`
  - 标定 UI：
    - 指导用户让振镜打标点阵
    - 自动识别点阵像素坐标
    - 计算并保存映射矩阵

### 1.2 语音 Agent 预留接口

- `edge/common/voice_agent/`（可后续实现）
  - KWS："Hi 香橙派" 本地唤醒
  - ASR：轻量中文命令识别
  - NLU：规则解析命令 → 结构化意图
  - 与业务层接口：
    - 获取最近一次检测结果（JSON）
    - 触发打标任务
    - 通过 TTS 播报结果

---

## 2. 路线 A：电子元器件智能识别与激光可视分拣

### 2.1 场景与目标

- 场景：桌面/托盘上随机摆放多种电子元件（包括色环电阻），顶视相机采集图像。
- 目标：
  - 使用 MindSpore YOLO 模型实时检测所有元件的位置和类别；
  - 对不同类别的元件使用不同激光图形进行可视标注；
  - 支持语音命令（可选）：例如“标出处台所有电阻”。

### 2.2 数据与模型

#### 2.2.1 数据集

- 公共数据集：
  - ElectroCom61 等电子元器件检测数据集（多类元件，带 bbox 标注）。
- 自采数据：
  - 使用项目实际元件和托盘拍摄 RGB 图像；
  - 使用 LabelImg/LabelMe 标注：resistor, capacitor, diode, transistor, ic, led, other 等。
  - 若区分色环电阻，可新增类别：`color_resistor`。

#### 2.2.2 模型选择

- 检测模型：
  - 基于 MindYOLO 的 YOLOv5n / YOLOv8n（或轻量 YOLOv7-tiny 等）
  - 输入尺寸：640×640
  - 类别数：约 6–10 类

- 可选扩展：色环电阻阻值识别
  - 增加一个小型分割/分类模型，对 YOLO 检测出的电阻 patch 进行色环解析。

### 2.3 训练端流程（training/route_a_electro）

1. 数据准备脚本
   - `prepare_electrocom61.py`：下载并转换公共数据集为 YOLO 格式
   - `prepare_custom_dataset.py`：将自采数据整理为统一格式

2. 训练脚本
   - `train_yolo_route_a.py`：
     - 加载数据集与配置
     - MindYOLO + MindSpore 训练
     - 支持 resume / fine-tune

3. 导出与转换
   - `export_mindir_route_a.py`：导出 MindIR
   - `convert_to_ascend_om.sh`：调用 ATC 将 MindIR → OM / Lite 模型

4. 评估
   - `eval_route_a.py`：计算 mAP、混淆矩阵

### 2.4 端侧应用（edge/route_a_app）

#### 2.4.1 运行流程

1. 初始化：
   - 加载标定矩阵、模型、硬件配置
2. 循环：
   - 采集 RGB(+Depth) 帧
   - 前处理 → MindSpore 推理
   - 后处理：NMS、类别过滤
   - 像素坐标 → 振镜坐标
   - 根据业务规则生成激光图形批次
   - 发送给振镜控制器执行
   - 将检测结果发布到 Web UI 与语音 agent

#### 2.4.2 类别与图形映射示例

- resistor/color_resistor：矩形框
- capacitor：圆
- ic：带方向矩形（在一端加短线表示引脚 1 方向）
- diode：箭头或小三角
- unknown/other：闪烁三角形

#### 2.4.3 depth 使用建议

- 工作平面平面拟合：
  - 每次启动时，用 depth 采集若干帧，对台面做 RANSAC 拟合，得到平面方程
  - 用于高度补偿和异常检测

- 堆叠/倾倒检测：
  - 若某个 bbox 内 Z 方差过大，则标记为“异常元件”，使用特殊激光图形提示

### 2.5 测试与演示

- 单元测试：
  - 校验模型推理接口
  - 校验坐标变换与标定精度
- 集成测试：
  - 在真实硬件上验证端到端延迟（目标 < 100ms/帧）
  - 测量激光标注位置误差（mm）

- Demo 场景脚本：
  - 预置多种元件在托盘中
  - 启动应用，展示：
    - 实时检测画面
    - 实物上的激光标注
  - 可选语音交互：“Hi 香橙派，标出处台所有电阻。”

---

## 3. 本地语音 Agent 设计概览

> 注：可作为后续版本实现，不影响主线功能。

### 3.1 模块划分

- `voice_agent/kws/`：唤醒词识别（Hi 香橙派）
- `voice_agent/asr/`：端侧语音识别（命令级语音）
- `voice_agent/nlu/`：规则/轻量模型解析意图
- `voice_agent/tts/`：文本转语音
- `voice_agent/bridge/`：与视觉/激光业务模块通信

### 3.2 命令语法示例

- 查询类：
  - “当前有多少个电阻？”
  - “告诉我电容的数量。”
- 打标类：
  - “标出处台所有电阻。”
  - “高亮显示所有未知器件。”
- 综合类：
  - “Hi 香橙派，帮我检查这盘器件并标出异常。”

解析后的结构化意图示例：

```json
{
  "intent": "highlight_objects",
  "target_class": "resistor",
  "scope": "current_workspace"
}
```

### 3.3 与视觉模块对接

- 视觉模块定期发布最新检测结果到本地消息总线（如 ZeroMQ/Redis/topic）。
- 语音 agent 根据意图从消息中读取对应类别对象列表：

```json
{
  "timestamp": 1234567890,
  "objects": [
    {"id": 1, "class": "resistor", "pixel_box": [x1, y1, x2, y2], "galvo_pos": [gx, gy]},
    {"id": 2, "class": "capacitor", ...}
  ]
}
```

- 语音 agent 生成打标任务：

```json
{
  "action": "draw_shapes",
  "objects": [
    {"id": 1, "shape": "rectangle", "galvo_pos": [gx, gy], "size": [w, h]},
    ...
  ]
}
```

并发送给 `laser_controller` 执行，同时通过 TTS 播报结果。

---

## 5. 后续规划

1. 优先完成：
   - 路线 A 的 YOLO 检测模型训练与端侧部署
   - 标定工具与激光打标闭环
2. 然后逐步加入：
   - depth 平面拟合与堆叠检测
   - 异常检测（使用 MVTec AD 预训练模型）
3. 最后实现：
   - 本地语音 agent
   - 更复杂的业务逻辑（如与 PLC/机械臂联动）

> 本文件只做总体设计纲要，细节请参考 `docs/route-a-design.md`。
