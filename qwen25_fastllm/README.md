# Qwen2.5-1.5B 在 Orange Pi AI Pro (Ascend 310B) NPU 部署指引

本目录提供最快速可复现的“板端自行编译 fastllm + 下载 Qwen2.5-1.5B + NPU 推理”脚本。默认目标设备为 Orange Pi AI Pro (20T, Ascend 310B，24GB 内存)。

## 目录结构

- `install_fastllm.sh`：在板子上编译并安装 fastllm（启用 Ascend 后端）。
- `download_qwen25.py`：使用 ModelScope 拉取 Qwen2.5-1.5B-Instruct 模型到本地 `models/`。
- `run_qwen25_fastllm.py`：基于 fastllm_pytools 的最小可运行推理脚本（交互式）。
- `requirements.txt`：Python 依赖（不含 fastllm，fastllm 需源码编译）。

## 环境前置

1) 确认板卡驱动与 CANN 已就绪：

```bash
npu-smi info
```

看到 310B4 且状态 OK 即可。随后：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

2) Python 3.8/3.9/3.10 任一版本可用，pip 可正常工作。

## 一键编译安装 fastllm（板端执行）

```bash
cd ~/ICT/qwen25_fastllm
chmod +x install_fastllm.sh
./install_fastllm.sh
```

脚本做的事情：
- 安装编译依赖 `cmake g++ python3-dev git`（若已安装会跳过）。
- 克隆 fastllm 源码（depth=1）。
- `cmake .. -DUSE_ASCEND=ON && make -j$(nproc)`。
- 安装 Python 绑定 `pip3 install .`。

## 下载模型（板端执行）

```bash
cd ~/ICT/qwen25_fastllm
pip3 install -r requirements.txt
python3 download_qwen25.py
```

- 默认使用 ModelScope，模型会落在 `./models/qwen/Qwen2.5-1.5B-Instruct`。
- 如需 HuggingFace，修改脚本中的 `use_modelscope` 或环境变量 `QWEN_USE_HF=1`。

## 运行推理（板端执行）

```bash
cd ~/ICT/qwen25_fastllm
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python3 run_qwen25_fastllm.py
```

特性：
- 默认 `dtype="float16"`，可按需改为 `"int8"` 以进一步省内存。
- 交互模式，输入 `exit` 退出。
- 示例首轮问题已写在脚本中。

## 常见问题

- **提示算子不支持或找不到 Ascend 运行时**：确认已执行 `set_env.sh`，并检查 `/usr/local/Ascend` 路径是否存在。
- **首次加载慢**：模型大小约 3GB，首次解压/校验较慢；可提前运行下载脚本。
- **内存不足**：24GB 内存足够 float16 运行 1.5B。如只剩 8GB，建议改为 `dtype="int8"`。

## 后续可扩展

- 将 `model_path` 替换为 `Qwen2.5-3B` 或 `Llama-3.2-3B` 后同样可跑；若是 7B 需量化为 int4/int8。
- 如需 WebUI，可在此基础上加上 Gradio（参考 `qwen_0_5b_chat_webui.py` 的接口结构）。

