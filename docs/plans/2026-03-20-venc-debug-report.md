# Orange Pi AI Pro / Ascend 310B1 VENC 问题调研汇报

## 1. 报告目的

本文用于向后续 Pro 调研同学交接当前 VENC 问题的背景、板卡状态、已完成验证、关键证据和当前判断，避免重复走弯路。

当前结论先行：

- **问题不是单纯的 Python 导入或自编译 `venc_wrapper.so` 问题。**
- **官方 V1 ACL 样例、官方 V2/HIMPI 样例、项目内自写 wrapper 三条路径均可稳定复现 VENC 通道创建失败。**
- **最强证据已经落到内核驱动层：`drv_venc -> dvpp_prot_mem_map -> iommu_map failed -34`。**
- 因此当前更像是 **板级驱动 / BSP / IOMMU / DVPP 受保护内存映射** 问题，而不是上层调用姿势问题。

## 2. 板卡与系统当前情况

### 2.1 板卡

- 板卡：**Orange Pi AI Pro**
- NPU 芯片：**Ascend 310B1**
- 当前通过 `python3 -c "import acl; print(acl.get_soc_name())"` 确认：`Ascend310B1`

### 2.2 当前系统

- 系统：Ubuntu 22.04.5 LTS
- 内核：`Linux ict 5.10.0+ #32 SMP Thu Sep 25 17:54:23 CST 2025 aarch64`
- `npu-smi` 版本：`25.2.0`

### 2.3 NPU 当前状态

最近核验结果：

- `npu-smi info` 显示：
  - Chip: `310B1`
  - Health: **Alarm**
  - Temperature: `60C`
  - Memory Usage: `4260 / 23674 MB`

这个 `Health=Alarm` 没有被进一步解释为单一原因，但它与 VENC 失败同时存在，值得作为板级异常信号保留。

## 3. 软件环境与版本情况

### 3.1 系统自带 CANN

板端仍保留一套系统安装路径：

- 路径：`/usr/local/Ascend/ascend-toolkit/latest`
- 版本：**CANN 8.0.0**

### 3.2 手动补装的 CANN

为了排除版本问题，已在板端额外安装：

- 路径：`~/miniconda3/Ascend/cann-8.5.0`
- 组件：
  - `cann-toolkit 8.5.0`
  - `cann-310b-ops 8.5.0`

### 3.3 重要说明

在夜间自动化中，曾出现一类**环境污染问题**：

- 直接 `source ~/miniconda3/Ascend/cann/set_env.sh` 后，某些样例在运行时会先报：
  - `libplatform.so: undefined symbol ... protobuf ...`
- 这个问题已经通过收紧自动化环境变量策略绕开，不再把它当作 VENC 主问题。
- 也就是说，**“protobuf/libplatform 符号冲突”是自动化环境问题，不是本次 VENC 主故障。**

## 4. 已完成的验证工作

### 4.1 项目内自编译 wrapper 路线

验证对象：

- `~/ICT/pybind_venc/src/venc_wrap.cpp`
- `venc_wrapper.so`

已做动作：

- 修正了构建文件，使其优先使用当前激活的 `CANN 8.5.0`
- 重新编译 `venc_wrapper.so`
- 用 `ldd` 确认运行时确实链接到：
  - `~/miniconda3/Ascend/cann-8.5.0/lib64/libascendcl.so`
  - `~/miniconda3/Ascend/cann-8.5.0/lib64/libacl_dvpp.so`

运行结果：

- `VencSession(640, 480, 'H264_MAIN')` 稳定失败
- 日志签名：
  - `Create venc channel failed, error 707`
  - `Dvpp venc init acl resource failed, error 707`

附加现象：

- 失败后偶发：
  - `corrupted double-linked list`
  - Python `abort/core dump`

判断：

- 这更像失败后的资源释放/析构问题，是**次生噪声**
- **主问题仍然是创建 VENC 通道失败**

### 4.2 官方 V1 ACL 样例

验证样例：

- `samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc`
- `samples_source/samples/cplusplus/level2_simple_inference/0_data_process/venc_image`

处理情况：

- 旧样例默认构建脚本与当前 `CANN 8.5.0` 目录布局不完全匹配
- 已通过补充 `DDK_PATH / INSTALL_DIR / LIBRARY_PATH / LDFLAGS` 等环境变量，使其能在板端编过

运行结果：

- `venc_image/out/main 4`
  - 稳定报：`fail to create venc channel, errorCode = 507018`
- `venc/scripts` 下运行最小 ACL 样例
  - 运行态仍然失败
  - 销毁阶段可见：`aclvencDestroyChannel failed, aclRet = 507018`

判断：

- **V1 ACL 官方样例也无法跑通**
- 说明“项目自写 wrapper 有问题”这一怀疑已经大幅减弱

### 4.3 官方 V2 / HIMPI 样例

验证样例：

- `samples_source/samples/cplusplus/level1_single_api/7_dvpp/venc_sample`

处理情况：

- 构建脚本同样需要补环境变量适配当前 `CANN 8.5.0`
- 样例已能编过

运行结果：

- `hi_mpi_venc_create_chn [0] faild with 0xa008800c`

判断：

- **V2/HIMPI 路线也没有绕过问题**
- 说明问题不是单独卡在 V1 ACL 这一条 API 路线

## 5. 夜间无人值守探索结果

已实现并运行两条自动化链路：

### 5.1 板端 watchdog

主脚本：

- `tools/board_venc_watchdog.sh`

启动脚本：

- `start_venc_watchdog_remote.sh`

夜间运行窗口：

- 启动：`2026-03-20 02:29:09 CST`
- 截止：`2026-03-20 07:05:37 CST`

运行模式：

- 初始全量验证 1 次
- 随后每 10 分钟 quick loop 1 次
- 截止前共完成：
  - `Initial Pass` 1 轮
  - `Quick Loop` 27 轮

夜间汇总结果：

- `pybind`：持续 hard fail
- `venc_image`：持续 hard fail
- `venc` ACL sample：持续 hard fail
- `V2/HIMPI` sample：初始验证失败，后续 quick loop 省略
- **直到 07:00 前没有出现任何一次成功编码**

### 5.2 本地 heartbeat 监控

本地监控脚本：

- `watch_venc_watchdog_heartbeat.sh`

作用：

- 每 60 秒抓一次板端：
  - `STATE.env`
  - `watchdog_launch.log` 尾部
  - 进程存活状态

目的：

- 确保不是“板端脚本挂了没人知道”
- 确保能确认夜间任务持续跑到了截止时间

## 6. 关键错误与证据

### 6.1 wrapper 路线

项目内 pybind 路线稳定错误：

- `Create venc channel failed, error 707`

### 6.2 官方 V1 ACL 样例

- `fail to create venc channel, errorCode = 507018`

### 6.3 官方 V2/HIMPI 样例

- `hi_mpi_venc_create_chn [0] faild with 0xa008800c`

### 6.4 内核侧关键证据

多次 `dmesg` 中稳定出现：

- `dvpp_prot_mem_map ... iommu_map failed -34`
- `media_prot_mem_malloc ... failed`
- `h264e_create_chn ... alloc encoder node buffer failed`
- `h265e_create_chn ... Alloc encoder context failed`
- `venc_create_chn_by_type ... Error 0xa008800c`
- `venc_create_chn ... create chnl failed`

这条链说明：

1. 用户态已经成功进入 `/dev/venc`
2. `drv_venc` 确实开始创建编码通道
3. 在 **DVPP protected memory / IOMMU map** 阶段失败
4. 返回内核私有错误码 `0xa008800c`
5. 上层再包装成 `707` / `507018`

## 7. 当前判断

### 7.1 基本排除项

以下方向基本已经排除为主因：

- 单纯 Python 导入问题
- 单纯 `PYTHONPATH` 配置问题
- 单纯项目自编译 `venc_wrapper.so` 编坏
- “只有 V1 ACL 用法不对”
- “只要改成 V2/HIMPI 就能好”

### 7.2 当前最可信判断

当前最可信的根因方向是：

- **板级驱动 / BSP / 固件 / IOMMU / DVPP protected memory 配置问题**

更具体地说：

- 不是“系统里没有 VENC”
  - 因为 `/dev/venc` 存在
  - `drv_venc / drv_h264e / drv_h265e / drv_vedu` 均已加载
- 而是“VENC 在当前板级系统配置下无法成功申请/映射所需内存资源”

### 7.3 额外风险信号

仍有几个异常信号需要一并记住：

- `npu-smi` 的 `Health = Alarm`
- 内核出现 `TS exception 0xb406000d`
- `pybind` 失败后存在二次析构崩溃噪声

这些未必是根因，但都说明当前板子并不是一个“纯净正常”的媒体运行环境。

## 8. 建议 Pro 下一步重点调研方向

### 8.1 优先级最高：板级驱动/BSP/设备树

请优先对比以下内容：

- Orange Pi AI Pro 当前系统镜像对应的 BSP 版本
- 是否存在官方提供的媒体/VENC 补丁、升级说明或已知问题
- `/proc/device-tree/reserved-memory`
- `/proc/cmdline`
- IOMMU / CMA / DVPP protected memory 相关配置

目标：

- 查清楚当前镜像是否真的对 310B1 的 VENC 做了完整板级适配

### 8.2 固件与驱动匹配关系

请核对：

- 当前内核驱动版本
- 当前 NPU firmware 版本
- `CANN 8.5.0` 与板端驱动/firmware 是否官方匹配

目标：

- 判断是否存在“用户态升级到了 8.5，但板级驱动/firmware 仍停在较老组合”的不匹配问题

### 8.3 官方样例与官方推荐路径

请继续确认：

- 针对 310B1 / Atlas 200I DK A2，官方当前到底推荐哪条 VENC 路径：
  - V1 ACL
  - V2/HIMPI
  - 或者专门的媒体样例仓

目标：

- 明确“正确调用方案”与“当前板子系统不支持”之间的边界

### 8.4 错误码映射

请重点追：

- `507018`
- `0xa008800c`
- `iommu_map failed -34`
- `0xb406000d`

目标：

- 把它们映射到官方内部错误码语义或已知缺陷记录

## 9. 可直接查看的本地材料

建议 Pro 直接从这些文件开始看：

- 夜间任务摘要：
  - `board_logs/venc_watchdog_20260320_022958/SUMMARY.md`
- 夜间完整值守日志：
  - `board_logs/venc_watchdog_20260320_022958/watchdog.log`
- 三条关键失败样例：
  - `board_logs/venc_watchdog_20260320_022958/01_pybind_probe/command.log`
  - `board_logs/venc_watchdog_20260320_022958/03_run_venc_image/command.log`
  - `board_logs/venc_watchdog_20260320_022958/07_run_v2_himpi/command.log`
- 板端 watchdog 脚本：
  - `tools/board_venc_watchdog.sh`
- 本地 heartbeat 监控：
  - `watch_venc_watchdog_heartbeat.sh`

## 10. 一句话总结

**当前板卡在项目自写 wrapper、官方 V1 ACL 样例、官方 V2/HIMPI 样例三条路径上都无法成功创建 VENC 通道；最强证据已经落到内核 `drv_venc -> dvpp_prot_mem_map -> iommu_map failed -34`，因此当前问题应按板级驱动/BSP/IOMMU/DVPP 资源映射问题继续深挖，而不是继续在上层调用代码里反复试错。**
