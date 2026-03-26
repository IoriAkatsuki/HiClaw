# Qwen3.5-4B 本机 ATC 转换与回传计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在本机 x86_64 上完成 Qwen3.5-4B patched ONNX 的 ATC 转换，并将生成的 OM 回传到板卡做基础验证。

**Architecture:** 复用板卡导出的 patched ONNX 与 external data，不在本机重新导出模型；本机仅承担 CANN/ATC 安装与 OM 编译，再把产物上传到板卡独立路径，避免干扰板卡现有长跑任务。若本机 CANN 与板卡运行时版本不一致，则以板卡 ACL smoke 结果为准判断兼容性。

**Tech Stack:** Conda/Mamba, Ascend CANN Toolkit, Ascend 310B ops, rsync/scp, ATC, ACL smoke。

---

### Task 1: 固定输入与输出路径

**Files:**
- Create: `docs/plans/2026-03-06-local-qwen35-atc-upload.md`
- Verify: `remote_sync/autocheck_qwen35_board.sh`

**Step 1: 记录板卡输入目录**
Run: `ssh HwHiAiUser@192.168.5.13 'ls -lah /home/HwHiAiUser/ICT/qwen35_autocheck/20260306_180158/onnx | sed -n "1,20p"'`
Expected: 看到 `qwen35_4b_seq128.atc.onnx` 与 external data 文件。

**Step 2: 记录板卡目标上传路径**
Run: `ssh HwHiAiUser@192.168.5.13 'mkdir -p /home/HwHiAiUser/ICT/x86_atc_uploads'`
Expected: 上传目录可用。

### Task 2: 准备本机 CANN 环境

**Files:**
- Create: `remote_sync/local_atc_qwen35_x86.sh`

**Step 1: 创建 Conda 环境并安装最小组件**
Run: `mamba create -n cann85-atc -c https://repo.huaweicloud.com/ascend/repos/conda python=3.10 ascend-cann-toolkit=8.5.0 ascend-cann-310b-ops=8.5.0 -y`
Expected: 环境创建成功，包含 toolkit 与 310b ops。

**Step 2: 验证 atc 可执行**
Run: `conda run -n cann85-atc bash -lc 'source "$CONDA_PREFIX"/Ascend/ascend-toolkit/set_env.sh && atc --version'`
Expected: 输出 ATC 版本信息。

### Task 3: 同步 ONNX external data

**Files:**
- Create: `local_qwen35_atc/`

**Step 1: 拉取 patched ONNX 目录**
Run: `rsync -a HwHiAiUser@192.168.5.13:/home/HwHiAiUser/ICT/qwen35_autocheck/20260306_180158/onnx/ local_qwen35_atc/onnx/`
Expected: 本地拿到 `qwen35_4b_seq128.atc.onnx` 和所有 external data 文件。

### Task 4: 本机执行 ATC

**Files:**
- Create: `remote_sync/local_atc_qwen35_x86.sh`
- Output: `local_qwen35_atc/out/qwen35_4b_seq128_fp16.om`

**Step 1: 写脚本固化环境与参数**
Run: `bash remote_sync/local_atc_qwen35_x86.sh`
Expected: 生成本机日志和 `.om` 文件。

### Task 5: 回传并验证

**Files:**
- Verify: `remote_sync/acl_smoke_qwen35.py`

**Step 1: 上传 OM 到板卡独立目录**
Run: `scp local_qwen35_atc/out/qwen35_4b_seq128_fp16.om HwHiAiUser@192.168.5.13:/home/HwHiAiUser/ICT/x86_atc_uploads/`
Expected: 板卡上能看到上传产物。

**Step 2: 在板卡做 ACL smoke**
Run: `ssh HwHiAiUser@192.168.5.13 'python3 /home/HwHiAiUser/ICT/remote_sync/acl_smoke_qwen35.py --model-path /home/HwHiAiUser/ICT/x86_atc_uploads/qwen35_4b_seq128_fp16.om --tokenizer-path /home/HwHiAiUser/ICT/models/qwen3.5-4b --max-seq-len 128 --max-new-tokens 8 --prompt "你好，请用一句话介绍香橙派 AI Pro。"'`
Expected: 成功输出文本，或明确暴露版本兼容错误。
