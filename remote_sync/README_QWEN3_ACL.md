# Qwen3-8B ACL 部署指南

## OrangePi AIpro (Ascend 310B) NPU加速部署

> 说明
>
> 本文档覆盖的是 `Qwen3` 一类标准 `Transformer` 模型在 `ACL / OM` 路线下的实验与部署方法，不应外推为“所有 `Qwen3.x / Qwen3.5` 模型都能在 `OrangePi AIpro (Ascend 310B1)` 上直接跑通 NPU 推理”。
>
> 对当前项目里的 `Qwen3.5` 而言，已确认 `Transformers + torch_npu` 路径会在首次前向挂起；同时也没有证据表明该混合 `Mamba-Attention` 架构已经被 `MindIE / ATB` 官方适配到 `310B1`。如果切换到 `ModelZoo` 中已明确适配 `OrangePi / 300I` 的模型，则 `MindIE / ATB` 仍然是值得尝试的 NPU 路线。

---

## 📋 目录

1. [环境要求](#环境要求)
2. [快速开始](#快速开始)
3. [详细步骤](#详细步骤)
4. [文件说明](#文件说明)
5. [故障排查](#故障排查)

---

## 🔧 环境要求

- **硬件**: OrangePi AIpro (Ascend 310B)
- **内存**: 24GB
- **系统**: Linux with Ascend CANN toolkit
- **Python**: 3.8+
- **依赖**:
  - transformers
  - huggingface_hub
  - streamlit
  - numpy
  - torch
  - acl (Ascend Computing Language)

---

## 🚀 快速开始

### 一键自动化部署

```bash
cd /home/HwHiAiUser/ICT
bash deploy_qwen3_acl.sh
```

这个脚本会自动完成：
1. ✓ 检查模型文件
2. ✓ 导出ONNX格式
3. ✓ 转换为OM格式（Ascend离线模型）
4. ✓ 测试ACL推理
5. ✓ 启动Web UI

---

## 📖 详细步骤

### 步骤 1: 下载Qwen3-8B模型

```bash
python3 download_qwen3_8b.py
```

**说明**: 从HuggingFace下载Qwen3-8B基础模型（约16GB）

**输出位置**: `/home/HwHiAiUser/ICT/models/qwen3-8b/`

---

### 步骤 2: 导出ONNX格式

```bash
python3 export_qwen3_onnx.py
```

**说明**: 将PyTorch模型转换为ONNX中间格式

**配置**:
- Batch size: 1
- Sequence length: 512
- Precision: FP16

**输出位置**: `/home/HwHiAiUser/ICT/qwen3_onnx/qwen3_model.onnx`

**预计耗时**: 5-15分钟（取决于CPU性能）

---

### 步骤 3: 转换为OM格式

```bash
bash convert_qwen3_to_om.sh
```

**说明**: 使用ATC工具将ONNX转换为Ascend离线模型

**技术细节**:
- Framework: ONNX (5)
- SOC Version: Ascend310B1
- Precision mode: allow_fp32_to_fp16
- Optimization: high_performance

**输出位置**: `/home/HwHiAiUser/ICT/qwen3_fp16.om`

**预计耗时**: 10-30分钟

---

### 步骤 4: 测试ACL推理

```bash
python3 qwen3_acl_inference.py
```

**说明**: 验证NPU推理功能

**测试内容**:
- ACL环境初始化
- 模型加载
- 3个测试prompt推理
- 性能benchmark

---

### 步骤 5: 启动Web UI

```bash
streamlit run qwen3_webui_acl.py --server.port 8501 --server.address 0.0.0.0
```

**访问地址**: `http://<板卡IP>:8501`

**或后台启动**:
```bash
nohup streamlit run qwen3_webui_acl.py --server.port 8501 --server.address 0.0.0.0 > webui_acl.log 2>&1 &
```

---

## 📁 文件说明

### 核心脚本

| 文件名 | 功能 | 说明 |
|--------|------|------|
| `download_qwen3_8b.py` | 模型下载 | 从HF下载Qwen3-8B |
| `export_qwen3_onnx.py` | ONNX导出 | PyTorch → ONNX |
| `convert_qwen3_to_om.sh` | OM转换 | ONNX → OM (ATC) |
| `qwen3_acl_inference.py` | ACL推理引擎 | NPU推理核心 |
| `qwen3_webui_acl.py` | Web界面 | Streamlit UI |
| `deploy_qwen3_acl.sh` | 一键部署 | 自动化脚本 |

### 模型文件

| 路径 | 内容 | 大小 |
|------|------|------|
| `/home/HwHiAiUser/ICT/models/qwen3-8b/` | 原始模型 | ~16GB |
| `/home/HwHiAiUser/ICT/qwen3_onnx/` | ONNX模型 | ~16GB |
| `/home/HwHiAiUser/ICT/qwen3_fp16.om` | Ascend OM | ~8GB |

---

## 🔍 故障排查

### 问题1: 下载失败

```bash
ERROR: Failed to download Qwen/Qwen3-8B
```

**解决方案**:
1. 检查网络连接
2. 验证HuggingFace token（如果需要）
3. 尝试使用镜像站点

### 问题2: ONNX导出内存不足

```bash
RuntimeError: CUDA out of memory
```

**解决方案**:
- 在脚本中已使用 `device_map="cpu"` 和 FP16
- 如仍失败，尝试减小 `SEQ_LENGTH`（当前512）

### 问题3: ATC转换失败

```bash
ATC run failed
```

**解决方案**:
1. 检查环境变量:
   ```bash
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```
2. 查看详细日志:
   ```bash
   ls -lh /var/log/npu/
   ```

### 问题4: ACL推理错误

```bash
acl.mdl.load_from_file failed
```

**解决方案**:
1. 验证OM文件存在:
   ```bash
   ls -lh /home/HwHiAiUser/ICT/qwen3_fp16.om
   ```
2. 检查NPU设备:
   ```bash
   npu-smi info
   ```

### 问题5: Web UI无法访问

**解决方案**:
1. 检查进程:
   ```bash
   ps aux | grep streamlit
   ```
2. 查看日志:
   ```bash
   tail -f webui_acl.log
   ```
3. 检查端口:
   ```bash
   netstat -tuln | grep 8501
   ```

---

## 📊 性能参考

### 推理性能（单token生成）

- **输入长度**: 256 tokens
- **推理时间**: ~0.5-1.0s (NPU)
- **内存占用**: ~8GB (模型) + ~4GB (运行时)

### 优化方向

1. **KV Cache**: 实现KV缓存以支持连续生成
2. **Dynamic Batch**: 支持动态batch以提高吞吐量
3. **混合精度**: 尝试INT8量化进一步压缩

---

## 📝 注意事项

### 当前限制

1. **单token生成**: 当前实现仅支持生成下一个token
2. **无KV Cache**: 每次推理需要重新计算整个序列
3. **固定长度**: 输入长度固定为512 tokens

### 后续改进

1. 实现完整的autoregressive生成
2. 添加KV Cache管理
3. 支持流式输出
4. 优化tokenizer集成

---

## 🤝 技术支持

遇到问题？

1. 查看日志文件
2. 检查Ascend CANN版本兼容性
3. 参考华为Ascend官方文档: https://www.hiascend.com/

---

## 📄 许可

遵循Qwen模型和Ascend toolkit的相关许可协议。

---

**最后更新**: 2025-12-06
**版本**: 1.0
**作者**: Claude Code
