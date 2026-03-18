# 振镜工作代码总览

本文档用于快速区分仓库中的振镜相关“上位机代码”和“单片机板卡驱动代码”，避免继续把示例、补丁、运行时输出混成一团。

## 1. 上位机代码（Ascend 板卡 / Linux 主机）

这部分负责视觉检测、标定、串口协议生成和调度。

### 核心目录

- `edge/laser_galvo/galvo_controller.py`
  - 上位机振镜控制器
  - 负责像素坐标到振镜坐标转换
  - 负责发送 `R/C/U` 绘图命令
- `edge/laser_galvo/calibrate_galvo.py`
  - 历史标定实现
  - 旧版依赖 `G/L` 命令；在当前正式固件协议下应视为历史参考
- `edge/laser_galvo/auto_calibrate.py`
  - 自动化校准封装
  - 负责串口探测、重试策略、诊断文件输出
- `edge/laser_galvo/run_calibration.sh`
  - 一键校准脚本
- `edge/unified_app/unified_monitor.py`
  - 统一检测入口
  - 在目标检测和安全监测基础上调用 `LaserGalvoController`

### 上位机验证脚本

- `tests/test_galvo_controller_compat.py`
  - 校验控制器对旧调用参数的兼容性
- `tests/test_calibration_quality.py`
  - 校验标定器的稳健聚合、质量门限和单应矩阵计算
- `tests/board_test_protocol.py`
  - 板级人工验证串口协议
- `tests/board_test_dac_range.py`
  - 板级人工扫描振镜物理范围
- `tests/board_test_visual_verify.py`
  - 板级人工验证激光点位响应

## 2. 单片机板卡驱动代码（STM32F401 + DAC8563）

这部分负责串口收命令、驱动 DAC、控制激光开关和执行轨迹。

### 正式工作代码

- `mirror/stm32f401ccu6_dac8563/Core/Src/main.c`
  - 当前正式协议入口
  - 已支持新的 `idx+C/R` 批量任务协议与 `U;` 提交
- `mirror/stm32f401ccu6_dac8563/HARDWARE/dac8563/dac8563.c`
  - DAC8563 底层驱动
- `mirror/stm32f401ccu6_dac8563/HARDWARE/dac8563/dac8563.h`
  - DAC8563 驱动头文件
- `mirror/stm32f401ccu6_dac8563/Core/Src/spi.c`
  - SPI 初始化
- `mirror/stm32f401ccu6_dac8563/Core/Src/usart.c`
  - 串口初始化
- `mirror/stm32f401ccu6_dac8563/Core/Src/gpio.c`
  - DAC 和激光相关 GPIO 初始化

### 补丁/参考文件

- `mirror/stm32_firmware_patch.c`
  - 历史补丁参考
  - 当前关键逻辑已经并入 `Core/Src/main.c`
- `edge/laser_galvo/stm32_galvo_protocol.c`
  - 独立协议示例
  - 属于早期二进制协议参考，不是当前正式工作链路

## 3. 当前正式工作链路

### 当前正式模型来源

1. 训练源：`2026_3_12/runs/train/yolo26n_aug_full_8419_gpu/weights/best.pt`
2. 正式部署产物：`models/route_a_yolo26/yolo26n_aug_full_8419_gpu.om`
3. 正式运行配置：`config/yolo26_6cls.yaml`
4. 导出入口：`tools/export_latest_yolo26_to_om.sh`

### 标定链路

1. 当前正式固件已切换到 `idx + 命令字母 + 分号分隔` 的批量协议
2. 正式命令形态为 `0C,...;1R,...;U;`
3. 旧版 `G/L` 校准命令不再属于当前正式固件协议
4. 仓库内 `auto_calibrate.py` / `calibrate_galvo.py` 目前应视为历史校准链路参考
5. 如要恢复自动校准，需要按新固件重新设计板端即时定位协议

### 正常绘制链路

1. `unified_monitor.py` 检测目标框
2. `galvo_controller.py` 将像素框转换为振镜坐标
3. 上位机批量发送 `0R,...;1C,...;U;`
4. STM32 在任务缓冲区执行绘制

## 4. 修改入口建议

### 如果你要改上位机行为

- 改坐标转换：`edge/laser_galvo/galvo_controller.py`
- 改标定算法：`edge/laser_galvo/calibrate_galvo.py`
- 改自动校准流程：`edge/laser_galvo/auto_calibrate.py`
- 改检测后联动：`edge/unified_app/unified_monitor.py`

### 如果你要改板卡协议或驱动

- 改串口命令解析：`mirror/stm32f401ccu6_dac8563/Core/Src/main.c`
- 改 DAC 输出逻辑：`mirror/stm32f401ccu6_dac8563/HARDWARE/dac8563/dac8563.c`
- 改引脚定义：`mirror/stm32f401ccu6_dac8563/Core/Src/gpio.c`

## 5. 本次整理后的约定

- `edge/laser_galvo/galvo_calibration.yaml` 属于运行时生成物，不再作为仓库源码提交
- `edge/laser_galvo/calibration_diagnostic*.json` 属于诊断输出，不作为源码提交
- 固件正式工作版本以 `mirror/stm32f401ccu6_dac8563/` 为准
- 上位机正式工作版本以 `edge/laser_galvo/` + `edge/unified_app/` 为准
