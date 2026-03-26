# 2026-03-11 D435 Root 目录迁移与实时代码评审

## 已验证现状

- 本机到板卡 `ict.local` 的 SSH 连接正常，`HwHiAiUser` 免密可登录。
- 远端项目根目录确认是 `/home/HwHiAiUser/ICT`。
- `HwHiAiUser` 对 `/root` 没有读取权限，直接访问返回 `Permission denied`。
- 远端 `sudo` 可用，但当前会要求密码；`root@ict.local` 直接 SSH 登录失败。
- 板卡当前用户目录里已经存在一套 D435/Route B/Unified/Laser 相关实时工作副本，关键路径包括：
  - `/home/HwHiAiUser/ICT/edge/route_b_app`
  - `/home/HwHiAiUser/ICT/edge/unified_app`
  - `/home/HwHiAiUser/ICT/edge/laser_galvo`
  - `/home/HwHiAiUser/ICT/start_route_b.sh`
  - `/home/HwHiAiUser/ICT/hand_yolov8n_pseudo_v1.om`

## Root 迁移结果

在拿到 sudo 密码后，已完成 `/root` 下候选目录的核查与迁移：

- `/root` 下唯一明确的 D435 相关项目目录是 `/root/d435_project`
- 该目录已完整迁移到 `/home/HwHiAiUser/ICT/d435_project`
- 迁移方式采用 `rsync` 复制校验后再删除源目录，不是直接 `mv`
- 迁移后已执行：
  - 源目录文件数 vs 目标目录文件数比对
  - `diff -rq /root/d435_project /home/HwHiAiUser/ICT/d435_project`
  - `chown -R HwHiAiUser:HwHiAiUser /home/HwHiAiUser/ICT/d435_project`
  - 删除 `/root/d435_project`

迁移后的目录包含 20 个文件，主要由以下几类组成：

- MediaPipe + RealSense 演示脚本：`run_d435.py`、`V1.py`、`V2.py`、`V3.py`、`V3_debug.py`
- YOLO / NPU 推理与推流脚本：`yolo_web_stream.py`、`npu_yolo_stream.py`、`run_yolo26.py`、`run_new.py`
- 检测 + 手部融合实验脚本：`combined.py`、`combinedV2.py`、`exp_yolo_mp.py`
- 模型与实验产物：`best.pt`、`best.onnx`、`best26.onnx`、`best26_npu.om`、`best_yolov8_npu.om`、`fusion_result.json`

## 已完成的远端实时取证

### Route B

- 远端实时 `route_b_app` 与当前仓库中的 `edge/route_b_app` 主体代码一致：
  - `hand_safety_monitor.py`
  - `hand_safety_monitor_mediapipe.py`
  - `webui_server.py`
  - `README.md`
- 已验证 `sha256` 完全一致。
- 唯一差异是远端实时目录多了一个 `edge/route_b_app/index.html`，但其内容与本地 `webui_http_safety/index.html` 完全一致，属于重复副本。
- 当前 `edge/route_b_app/webui_server.py` 固定服务目录是 `~/ICT/webui_http_safety`，不会读取 `route_b_app/index.html`，因此这个重复副本不应当作为“优点”吸收。

### Unified

- 远端实时 `edge/unified_app/unified_monitor.py` 相比当前仓库，保留了两类值得单独评估的小思路：
  - `acl_ret_code()`：兼容 ACL Python 绑定返回 `int` 或 `tuple`
  - `--hand-model` / `--pose-model` 双来源切换：专用 hand detector 与 pose fallback 二选一
- 但当前仓库版本在主流程上更收敛：
  - 只保留显式 `--pose-model`
  - 去掉 hand detector 并行路径
  - 状态统计、CLI 参数和执行分支更简单
- 结论：不应回退远端整段实现，只值得把 `acl_ret_code()` 这种低成本兼容保护抽出来评估是否补回。

### Laser Galvo

- 当前仓库 `edge/laser_galvo` 整体明显优于远端实时版本。
- 当前仓库新增或增强了：
  - 更稳健的光斑检测与筛选
  - 多次采样与聚合
  - 质量门限
  - 结构化诊断输出
  - RealSense/USB 抽象
  - 文档与脚本收敛
- 远端实时目录里额外存在 `simple_calibrate.py`、`green_calibrate.py`、`diagnose.py`、`capture_test.py`、`test_laser.py` 这类工具脚本，但它们的共同问题是：
  - 硬编码 `/home/HwHiAiUser/ICT/...`
  - 交互式或一次性脚本风格明显
  - 对颜色、环境光和设备状态假设过强
- 结论：这些脚本只有“快速排障思路”可参考，不建议直接并回当前主干。

## 相比当前代码，可取之处

### 建议吸收

1. `edge/unified_app/unified_monitor.py` 里的 ACL 返回码兼容层
   - 价值：防御不同 ACL Python 绑定返回格式差异
   - 成本：极低
   - 风险：低

2. “专用 hand detector 作为可选增强通道”的产品思路
   - 价值：在复杂遮挡或姿态点不稳时，给统一监控一个备用感知路径
   - 约束：必须作为可选特性独立实现，不能把当前主流程回退成双分支混跑

3. “极简自举部署”的经验
   - 远端实时版更偏向板卡脏环境下先跑起来
   - 适合做成单独的 fallback 脚本或恢复脚本，不适合污染正式主入口

4. `d435_project` 里的“采集 / 推理 / 展示解耦”思路
   - `combinedV2.py` 用采集线程、NPU 线程、MediaPipe 线程拆开处理，并用双端口分别暴露 NPU 视图与 MediaPipe 视图
   - 这说明学弟已经意识到单线程串行循环会把相机采集、NPU 推理、CPU 手部关键点和 HTTP 推流互相阻塞
   - 当前主线如果后续要恢复更复杂的多模型并发，这个解耦方向值得参考，但应基于现有 `unified_app` 重新设计，而不是直接拷脚本

5. `d435_project` 里的“YOLO 先检手框，再局部跑 MediaPipe”思路
   - `exp_yolo_mp.py` 先用 NPU 检出 hand bbox，再对裁剪 ROI 跑 MediaPipe，并在无新框时短暂复用上帧 bbox
   - 这个思路比全帧 MediaPipe 更接近真正的性能优化，尤其适合你后续要兼顾 NPU 检测和 CPU landmark 的场景
   - 它是一个有价值的实验方向，但当前实现仍是原型级，不应直接纳入主线

### 当前已更优，不应回退

1. `edge/laser_galvo/calibrate_galvo.py`
2. `edge/laser_galvo/auto_calibrate.py`
3. `edge/laser_galvo/galvo_controller.py`
4. `edge/unified_app/unified_monitor.py` 的单主路径收敛
5. `edge/unified_app/webui_server.py` 与 `webui_http_unified/index.html` 的职责分离

### 不建议吸收

1. 远端 `route_b_app/index.html`
   - 原因：与 `webui_http_safety/index.html` 重复，且当前服务端不读取它

2. 远端 `unified_app/webui_server.py` 的内嵌整页 HTML
   - 原因：适合临时自举，不适合长期维护

3. 远端 `laser_galvo` 的旧式单帧绿色阈值检测逻辑
   - 原因：鲁棒性明显弱于当前版本

4. 远端 `unified_app` 的 hand/pose 自动 fallback 主入口
   - 原因：会重新引入并行推理路径和配置歧义

5. `d435_project` 里的大多数脚本
   - 原因：原型味很重，普遍存在硬编码模型名、硬编码板卡 IP、硬编码 `/root/d435_project` 路径、无 CLI、无配置层、无测试、重复代码多的问题
   - 例如：
     - `exp_yolo_mp.py` 仍写死 `MODEL_PATH = "/root/d435_project/best26_npu.om"`
     - `combined.py`、`combinedV2.py`、`run_yolo26.py` 等脚本直接把访问地址写成 `http://192.168.5.13:8080`
     - 多个脚本都各自复制了一套 `NpuYoloEngine`、HTTP 推流和 RealSense 初始化逻辑

## d435_project 专项评审

### 真正值得吸收的思路

1. 分线程解耦
   - 依据：`/home/HwHiAiUser/ICT/d435_project/combinedV2.py`
   - 特征：采集线程独立，NPU 推理独立，MediaPipe 推理独立，展示端口独立
   - 价值：这是对“采集阻塞推理、推理阻塞推流”的真实工程问题的有效回应

2. ROI 级 MediaPipe
   - 依据：`/home/HwHiAiUser/ICT/d435_project/exp_yolo_mp.py`
   - 特征：先 hand bbox，再 `crop_with_pad()` 后局部 landmark
   - 价值：相比全帧手部关键点更接近正确的性能优化路径

3. 快速自举式板卡调试
   - 依据：`run_d435.py`、`V1.py` ~ `V3_debug.py`
   - 特征：最小依赖、直出 FPS、直看深度值、直开 HTTP 流
   - 价值：适合快速确认 D435、MediaPipe、NPU 模型哪一层先坏

### 当前主线已明显更优

1. 工程化程度
   - 当前主线已有 `route_b_app`、`unified_app`、`laser_galvo` 的明确职责边界
   - `d435_project` 仍是脚本集合，缺少模块边界和配置治理

2. Web 展示模式
   - 当前主线统一写 `frame.jpg` + `state.json` 给 WebUI 消费
   - `d435_project` 多数脚本直接自带 MJPEG HTTP 服务，适合调试，不适合产品化

3. 标定与诊断
   - 当前 `edge/laser_galvo` 已有更成熟的质量门限、诊断文件和稳健聚合
   - `d435_project` 主要聚焦相机 + 检测，不具备同等级的标定工程能力

### 不建议直接吸收的原型实现

1. 任何硬编码 `/root/d435_project` 的路径
2. 任何硬编码 `192.168.5.13` 的输出提示或部署假设
3. 重复定义 `NpuYoloEngine`、HTTP Server、相机初始化的脚本式写法
4. 单脚本同时承担采集、推理、展示、异常处理的“大一统脚本”

## 拿到 sudo 后的执行命令模板

下面这些命令保留为复盘模板；本次实际迁移已经完成：

```bash
ssh -tt HwHiAiUser@ict.local '
TS=$(date +%Y%m%d_%H%M%S)
sudo find /root -maxdepth 4 \
  \( -iname "*d435*" -o -iname "*realsense*" -o -iname "*route_b*" -o -iname "*hand*" -o -iname "*laser*" -o -iname "*unified*" \) \
  -printf "%M %u:%g %s %TY-%Tm-%Td %TH:%TM %p\n" | tee /tmp/root_d435_inventory_${TS}.txt
'
```

确认清单后，再按实际目录逐项迁移，推荐先迁到独立暂存目录而不是直接覆盖：

```bash
ssh -tt HwHiAiUser@ict.local '
TS=$(date +%Y%m%d_%H%M%S)
TARGET=/home/HwHiAiUser/ICT/migrated_from_root_${TS}
sudo install -d -o HwHiAiUser -g HwHiAiUser "$TARGET"

# 例：把已经确认的某个目录搬到用户目录下的暂存区
sudo rsync -a /root/<确认后的源目录>/ "$TARGET"/<目标子目录>/
sudo chown -R HwHiAiUser:HwHiAiUser "$TARGET"
'
```

如果确认需要“移动而非复制”，应先完成一次复制校验，再单独删除源目录，避免误删：

```bash
ssh -tt HwHiAiUser@ict.local '
sudo rsync -a /root/<确认后的源目录>/ /home/HwHiAiUser/ICT/migrated_from_root_verified/<目标子目录>/
sudo chown -R HwHiAiUser:HwHiAiUser /home/HwHiAiUser/ICT/migrated_from_root_verified
# 人工确认无误后，再执行 sudo rm -rf /root/<确认后的源目录>
'
```

## 本次执行留下的本地证据

- 远端实时快照已同步到本机临时目录：
  - `/tmp/ict_remote_live/route_b_app`
  - `/tmp/ict_remote_live/unified_app`
  - `/tmp/ict_remote_live/laser_galvo`
  - `/tmp/ict_remote_live/d435_project`
- 这些快照用于本次差异审查，来源是：
  - `rsync -av -e ssh HwHiAiUser@ict.local:/home/HwHiAiUser/ICT/... /tmp/ict_remote_live/...`

## 建议的下一步

1. 如果要把 `d435_project` 里的实验价值进一步上收，优先评估：
   - 在正式架构里引入“采集 / 推理 / 展示解耦”的线程模型
   - 把“YOLO 先出 hand bbox，再局部跑 MediaPipe”的思路做成独立实验分支
2. 不建议把 `d435_project` 直接并入 `edge/` 主线；它更适合作为原型参考目录保留在 `/home/HwHiAiUser/ICT/d435_project`
3. 当前已经补入 `acl_ret_code()` 兼容层，可以继续观察板卡上是否还存在 ACL Python 绑定返回值差异问题
2. 拿到 `/root` 清单后，只迁 D435/Route B/Unified/Laser 相关目录，不做整包搬迁。
3. 如果你要继续吸收远端思路，优先评估把 `acl_ret_code()` 小补丁并入当前 `edge/unified_app/unified_monitor.py`。
