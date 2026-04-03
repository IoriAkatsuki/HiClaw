# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Laser-AR Sorting system on Orange Pi AI Pro (Ascend 310B NPU). Detects electronic components via YOLO, monitors hand safety via pose estimation, and projects laser annotations onto detected objects through a galvanometer mirror system. All Python, no build system — scripts run directly on the board.

## Hardware Context

- **Board**: Orange Pi AI Pro (Ascend 310B NPU, `soc_version=Ascend310B1`)
- **Camera**: Intel RealSense D435 (RGB + Depth, 640x480)
- **Galvo**: DAC8563 dual-channel via STM32F401CCU6 over UART (115200, 8N1)
- **Dev host**: Linux PC syncing code to board via rsync

## Key Commands

```bash
# Ascend environment (required before any NPU inference)
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Run unified detection (on board)
./start_unified.sh

# Model conversion (MindIR → OM for Ascend NPU)
bash training/route_a_electro/convert_to_ascend_om.sh <input.mindir> <output.om> [img_size=640]

# ONNX model conversion via ATC
atc --model=model.onnx --framework=5 --output=model --soc_version=Ascend310B1

# Sync code from board to local
./sync_from_board.sh   # rsync from HwHiAiUser@ict.local

# Run tests (standard unittest, no pytest)
python3 -m unittest discover -s tests -v

# Galvo calibration
cd edge/laser_galvo && python3 calibrate_galvo.py --serial-port /dev/ttyUSB0 --output galvo_calibration.yaml
```

## Architecture

### Three Application Routes

1. **Route A** (`edge/route_a_app/`): Electronic component detection. `pipeline.py` = FastAPI + MindSpore Lite inference. Uses `edge/common/ms_infer/yolov8_ms.py` for YOLO inference wrapper.
2. **Route B** (`edge/route_b_app/`): Hand safety monitoring. `hand_safety_monitor.py` = YOLOv8-Pose via ACL on NPU. `hand_safety_monitor_mediapipe.py` = CPU fallback via MediaPipe.
3. **Unified** (`edge/unified_app/`): Combines Route A + B + laser. `unified_monitor.py` = main entry, imports galvo controller from `edge/laser_galvo/`. `unified_monitor_with_laser.py` = laser-enabled variant.

### Shared Modules (`edge/common/`)

- `ms_infer/yolov8_ms.py` — MindSpore Lite YOLOv8 inference: load model, preprocess, NMS, class names from `data.yaml`
- `calibration/camera_galvo_mapping.py` — Homography-based pixel→galvo coordinate transform (`HomographyCalib` dataclass)
- `hw_config.yaml` — Camera intrinsics, workspace plane, galvo range, performance budget

### Laser Galvo System (`edge/laser_galvo/`)

- `galvo_controller.py` — `LaserGalvoController` class: serial connection, homography transform, `draw_box()` sends text commands to STM32
- `calibrate_galvo.py` — `GalvoCalibrator` class: automated calibration via laser spot detection (HSV + frame differencing), outputs YAML homography matrix
- `generate_calibration_target.py` — Generates printable A4 calibration board
- STM32 text protocol: `R<idx><x>,<y>,<w>,<h>` (rectangle), `C<idx><x>,<y>,<r>` (circle), `G<x>,<y>` (goto), `L<on/off>` (laser), `U` (update/execute)

### STM32 Firmware (`mirror/`)

- `stm32f401ccu6_dac8563/` — Full Keil MDK project for STM32F401CCU6 + DAC8563
- `HARDWARE/dac8563/` — DAC8563 SPI driver
- `HARDWARE/zzy_lib_uart_f4/` — UART library for command reception
- `stm32_firmware_patch.c` — Patch file documenting added G/L commands for `main.c`

### Training (`training/route_a_electro/`)

Pipeline: `prepare_electrocom61.py` → `train_yolo_route_a.py` → `export_mindir_route_a.py` → `convert_to_ascend_om.sh` (ATC). Dataset: ElectroCom-61 (61 electronic component classes).

### WebUI

Each app route serves a simple HTTP WebUI (frame.jpg + state.json polling):
- Route A: port 8001
- Unified: port 8002
- Route B: port 8003

`edge/*/webui_server.py` serves static files from `webui_http*/` directories.

## Code Patterns

- **NPU inference**: Two paths — MindSpore Lite (`ms_infer/yolov8_ms.py` for Route A) and raw ACL (`acl` module, inline `AclLiteResource` class duplicated in Route B and Unified). The ACL boilerplate is copy-pasted, not shared.
- **Optional imports**: Hardware-dependent modules (`acl`, `mediapipe`, `pyrealsense2`) are wrapped in `try/except` to allow running on dev machines.
- **Path resolution**: Scripts use `Path(__file__).resolve().parents[N]` and `sys.path.insert` to find sibling packages — no `setup.py` or installable packages.
- **Config format**: YAML for hardware config and model class names (`data.yaml`), YAML for calibration output (homography matrix).
- **Tests**: `unittest` based, in `tests/`. Tests stub hardware by subclassing controllers (e.g., `DummyLaserGalvoController` intercepts serial writes). Run with `python3 -m unittest discover -s tests -v`.
- **Board user**: `HwHiAiUser` on hostname `ict.local`. Dev sync target: `/home/HwHiAiUser/ICT/`.

## Important Constraints

- Model files (`.om`, `.onnx`, `.pt`, `.mindir`) are gitignored — never commit them.
- The Ascend environment (`set_env.sh`) must be sourced before any NPU code runs.
- Galvo coordinates are int16 (-32768 to 32767), mapped from pixel space via homography matrix. Calibration accuracy target: <1mm.
- Safety: laser auto-disables when hand distance < 300mm (configurable `--danger-distance`).
