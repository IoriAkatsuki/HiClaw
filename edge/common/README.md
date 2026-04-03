# edge/common 模块说明

共享基础能力：相机采集、MindSpore Lite 推理、标定、振镜控制、Web UI、语音接口。建议逐模块迭代开发并保留单元测试。

## 目录约定
- realsense_capture：封装 D435i，同步 RGB+Depth
- ms_infer：MindSpore Lite 推理包装，含预处理/后处理
- calibration：相机→物理→振镜坐标映射与标定工具
- laser_controller：串口/TCP 驱动振镜控制卡，上层图形接口
- ui_web：轻量 Web 可视化，展示画面、检测框、统计
- voice_agent：预留语音唤醒/ASR/NLU/TTS 对接

## 配置
- `hw_config.yaml`：相机内参、振镜范围、性能预算等；按硬件实测填写

## 测试建议
- 单元测试：推理封装、坐标变换、通信协议
- 集成测试：端到端延迟、打标精度、Web UI 实时性
