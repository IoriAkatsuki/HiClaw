# 项目移交说明（CV）

## 设备与目录
- 板卡：ict.local（用户 `HwHiAiUser`，已免密 SSH）。
- 板卡工作根：`~/ICT`
  - 模型权重：`runs/detect/train_electro61/weights/yolov8_electro61_aipp.om`
  - 数据集：`ElectroCom61 A Multiclass Dataset for Detection of Electronic Components/ElectroCom-61_v2/data.yaml`
  - WebUI 根目录：`webui_http`（`python3 -m http.server 8000 --bind 0.0.0.0` 已常驻）
  - 推流/推理脚本：`edge/route_a_app/rtsp_detect_aipp.py`
  - PyBind DVPP 编码器：`pybind_venc/`（`build/venc_wrapper.so`）
  - 日志示例：`rtsp_aipp.log`、`rtsp_detect.log`、`mediamtx.log`
- 本机（当前目录）：`/home/oasis/Documents/ICT`（含镜像/备份，不作为运行环境）。

## 已完成工作
1) **AIPP 推理管线**：摄像头 /dev/video4 -> AIPP YOLOv8 OM -> 后处理 -> WebUI 刷新。
   - 脚本：`~/ICT/edge/route_a_app/rtsp_detect_aipp.py`
   - 输出：`webui_http/frame.jpg`、`webui_http/state.json`（0.5s 刷新）。
   - 状态示例：`fps≈40+`、`infer_ms≈1ms`（无目标时 dets 为空）。
2) **模型信息**：输出缓冲大小 2,184,000 bytes，对应 shape [1,65,8400]（nc=61）。
3) **DVPP PyBind 封装**：`pybind_venc/build/venc_wrapper.so`（源自 `samples_source/samples/cplusplus/common/acllite`）。
   - 修改点：VencHelper 增加内存队列输出、空指针保护；AclLiteResource 允许 REPEAT_INITIALIZE；CMake 加 `-DENABLE_DVPP_INTERFACE`。
4) **服务常驻**：`http.server 8000`、`mediamtx_bin` 已常驻；推理脚本可通过 nohup 启动。

## 待解决/未完成
- **DVPP 硬编码失败**：VENC 创建报错 707 / Set eos ret=100000，导致 PyBind `VencSession` 初始化失败，目前脚本在 DVPP 失败时自动禁用硬编码，仅保留推理+WebUI。
- **硬编码路径未启用**：无 UDP 码流输出（`--out-ip/--out-port` 参数暂时无效）。

## 启动/重启步骤
```bash
# 进入板卡
ssh HwHiAiUser@ict.local
cd ~/ICT

# 启动推理（禁用 DVPP 容错，主要用于 WebUI 监看）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=~/ICT/pybind_venc/build:/usr/local/Ascend/ascend-toolkit/latest/python/site-packages:$PYTHONPATH
nohup python3 edge/route_a_app/rtsp_detect_aipp.py \
  --model runs/detect/train_electro61/weights/yolov8_electro61_aipp.om \
  --data-yaml "~/ICT/ElectroCom61 A Multiclass Dataset for Detection of Electronic Components/ElectroCom-61_v2/data.yaml" \
  --cam /dev/video4 \
  --out-ip 127.0.0.1 --out-port 8555 \
  > rtsp_aipp.log 2>&1 &

# 查看 WebUI
# 浏览器访问 http://ict.local:8000/ 读取 frame.jpg/state.json
```

## 诊断要点
- 模型输出 reshape：`pred = flat.reshape(1,65,8400).transpose(0,2,1)`；nc=61，conf=0.25，iou=0.45。
- AIPP 输入：直接喂 uint8 BGR（640x640 letterbox），不做 /255、转置。
- 摄像头：`/dev/video4`，已设置 640x480@30 + AUTO_EXPOSURE=0.75。

## 修复 DVPP 硬编码的建议
1) 用 root 运行官方 venc 样例确认 310B VENC 功能；检查 `/dev/davinci*` 权限与驱动版本。
2) 若需继续调试 PyBind：
   - 代码根：`pybind_venc/src/venc_wrap.cpp`（VencSession 封装）
   - 依赖源：`samples_source/samples/cplusplus/common/acllite/src/{VencHelper.cpp,AclLiteResource.cpp}`
   - CMake：`pybind_venc/CMakeLists.txt`（已链接 `ascendcl`、`acl_dvpp`）
3) 若 VENC 创建仍失败，考虑改用 ffmpeg+libx264 软编或 mediamtx 直接拉取 rawvideo。

## 快速查阅
- WebUI 状态：`cat ~/ICT/webui_http/state.json`
- 最近日志：`tail -n 50 ~/ICT/rtsp_aipp.log`
- 进程：`ps -ef | egrep 'rtsp_detect_aipp.py|http.server'`
