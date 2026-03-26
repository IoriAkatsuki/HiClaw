# Qwen3.5-0.8B 夜间 ATC 自动化说明

## 目标

在不升级固件的前提下，自动完成以下流程：

- 本机 conda 环境中的 `CANN 8.5.0` 工具链核验
- 从板卡发现并同步 `Qwen3.5-0.8B-ONNX`
- 按矩阵尝试 `atc` 编译
- 若产出 `.om`，自动回传板卡并执行最小 ACL smoke
- 在截止时间前生成最终 Markdown 报告

## 主要文件

- 主控脚本：`qwen35_nightly_atc.py`
- 后台启动脚本：`start_qwen35_nightly_atc.sh`
- 单元测试：`tests/test_qwen35_nightly_atc.py`

## 默认行为

- 目标板卡：`HwHiAiUser@ict.local`
- conda 环境名：`cann850-qwen35-nightly`
- 截止时间：次日 `09:00`
- 编译矩阵：
  - `seq_len=64,128`
  - `precision_mode=allow_fp32_to_fp16,must_keep_origin_dtype`
  - `onnx_variant=raw`
- 截止保护窗口：距离截止 `20` 分钟内不再启动新 case

## 推荐启动方式

```bash
cd /home/oasis/Documents/ICT
chmod +x start_qwen35_nightly_atc.sh
./start_qwen35_nightly_atc.sh
```

查看启动日志：

```bash
tail -f /home/oasis/Documents/ICT/nightly_qwen35_0p8b/latest_launcher.log
```

查看主运行目录：

```bash
readlink -f /home/oasis/Documents/ICT/nightly_qwen35_0p8b/latest_run
```

## 常用参数

指定截止时间：

```bash
./start_qwen35_nightly_atc.sh --deadline-at 09:00
```

指定远端 ONNX 路径：

```bash
./start_qwen35_nightly_atc.sh \
  --remote-onnx-path /home/HwHiAiUser/ICT/qwen35_0p8b_onnx/model.onnx
```

指定 tokenizer 路径：

```bash
./start_qwen35_nightly_atc.sh \
  --remote-tokenizer-dir /home/HwHiAiUser/ICT/models/qwen3.5-0.8b
```

跳过板端 smoke，仅验证交叉编译：

```bash
./start_qwen35_nightly_atc.sh --skip-board-smoke
```

仅演练命令而不真正执行：

```bash
./start_qwen35_nightly_atc.sh --dry-run
```

## 输出产物

每次运行会创建一个目录：

```text
nightly_qwen35_0p8b/<run_id>/
```

关键产物包括：

- `master.log`：主控日志
- `snapshots/env_snapshot.txt`：本机环境快照
- `snapshots/board_snapshot.txt`：板卡只读快照
- `snapshots/model_snapshot.txt`：模型和 ONNX 路径快照
- `matrix_summary.csv`：编译矩阵结果
- `nightly_final_report.md`：最终结论
- `runs/<case_id>/atc.log`：单 case ATC 日志
- `runs/<case_id>/meta.json`：单 case 结构化结果

## 结论判读

最终报告重点回答：

- 工具链是否就绪
- 模型是否仍包含 `linear_attention`
- 是否至少有一个 case 成功产出 `.om`
- 若失败，主阻塞点更像：
  - `RmsNorm/ResNorm`
  - 还是 `linear_attention / triangular_update / Mul`
- 是否进入板端 ACL smoke
