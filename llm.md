# OrangePi AIpro (Ascend 310B) LLM 部署项目状态报告

## 1. 项目目标
在拥有 24GB 内存的 OrangePi AIpro (Ascend 310B) 开发板上，部署并运行 Qwen 系列大语言模型，并提供 Web UI 交互界面。最终目标是实现利用 NPU 加速推理。

## 2. 已完成工作 (截至 2025年12月6日)

### 2.1 基础环境与 FastLLM 编译
- **FastLLM 源码编译**: 成功拉取并编译了 `ztxz16/fastllm` (master分支)。
  - *现状*: 尽管使用了 `-DUSE_ASCEND=1` 标志，但当前 master 分支似乎缺失原生的 Ascend 310B (ACL) 支持代码，因此 FastLLM 目前回退到 **CPU 模式**运行。
  - *Python 绑定*: 成功编译并安装了 Python 绑定 (`ftllm` / `pyfastllm`)，并解决了 `transformers` 等依赖问题。

### 2.2 模型准备
- **Qwen2.5-7B-Instruct**: 成功下载并验证。
- **Qwen3-8B-Instruct**: 尝试下载，但因 HuggingFace 权限/资源不存在问题失败。目前已配置自动回退机制，默认使用 Qwen2.5-7B。
- **FP16 高精度**: 利用板载 24GB 大内存优势，已在代码中强制使用 `float16` 精度加载模型，避免了量化带来的精度损失。

### 2.3 NPU (ACL) 推理验证 (Proof of Concept)
为了验证 310B 的 NPU 能力，我们绕过 FastLLM，直接使用华为官方工具链跑通了全流程：
1.  **ONNX 导出**: 编写脚本将 Qwen2.5 模型成功导出为 ONNX 格式（简化版，无 KV Cache）。
2.  **ATC 转换**: 使用 `atc` 工具将 ONNX 成功转换为 Ascend 离线模型 (`.om`)，精度 FP16。
3.  **ACL 推理**: 编写了 Python ACL 脚本 (`run_acl_inference.py`)，成功加载 `.om` 模型并在 NPU 上完成了一次推理（耗时 ~0.64s）。
    - *结论*: 硬件和驱动环境正常，具备运行 LLM 的能力，但手写完整的 ACL 推理引擎（含 KV Cache 和 Tokenizer）工程量巨大。

### 2.4 Web UI 开发
- **Streamlit 界面**: 开发了基于 `streamlit` 的 Web 聊天界面 (`run_qwen_webui_fixed.py`)。
- **功能**: 支持流式输出、历史对话记忆。
- **修复**: 解决了 Python 路径导致的 `ModuleNotFoundError` 和端口占用问题。
- **状态**: 脚本已上传至板卡，正处于后台启动运行阶段。

## 3. 当前待完成工作 (交接重点)

### 3.1 Web UI 服务验证
- **任务**: 确认 Web UI 是否在后台成功启动。
- **操作**: 
  1. SSH 登录板卡: `ssh HwHiAiUser@ict.local`
  2. 检查进程: `ps aux | grep streamlit`
  3. 检查日志: `cat ~/ICT/webui.log`
  4. 访问: 打开浏览器访问 `http://<板卡IP>:8501` 进行对话测试。

### 3.2 性能优化 (NPU 集成)
由于 FastLLM 当前分支不支持 NPU，而手写 ACL 太复杂，接下来的重点是寻找**中间件解决方案**：
- **方案 A (推荐)**: 寻找适配了 Ascend 的 FastLLM 分支（如 `fastllm-ascend` 或社区 fork）并重新编译。
- **方案 B**: 尝试华为官方的 **MindIE** (Mind Inference Engine) 框架，这是目前在 Ascend 上部署 LLM 的标准推荐方案，支持 transformer 结构和高并发。
- **方案 C**: 继续完善 ACL 推理脚本，增加 KV Cache 管理和 Tokenizer 集成（难度高，仅作为最后手段）。

### 3.3 模型升级
- **任务**: 持续关注 Qwen3-8B 的发布状态或权限要求。一旦可用，通过 `huggingface-cli` 下载并替换现有模型路径。

## 4. 关键文件路径
- **项目根目录**: `/home/HwHiAiUser/ICT`
- **Web UI 启动脚本**: `~/ICT/run_qwen_webui_fixed.py`
- **FastLLM 编译目录**: `~/ICT/qwen25_fastllm/fastllm_src/build`
- **模型路径**: `~/ICT/qwen25_fastllm/models/qwen/Qwen/Qwen2.5-7B-Instruct`
- **日志文件**: `~/ICT/webui.log`