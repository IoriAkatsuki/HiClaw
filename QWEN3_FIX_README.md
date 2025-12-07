# Qwen3-1.7B 乱码问题修复说明

## 问题诊断

原代码只生成**单个token**，导致以下问题：
1. **输出不完整** - 只返回下一个token，不是完整回复
2. **中文乱码** - 单个token可能是中文字符的一部分，导致UTF-8解码错误
3. **无法对话** - 没有实现自回归循环生成

## 修复内容

### 1. `qwen3_acl_inference.py` - 推理引擎修复

#### 新增方法：
- **`_inference_single_step()`** - 单步推理辅助方法
- **`generate_text()`** - 完整文本生成（新增主要功能）

#### 关键改进：
```python
def generate_text(self, prompt, max_new_tokens=128, temperature=1.0, top_k=50):
    """完整的自回归文本生成"""
    # 1. 应用chat模板格式化
    # 2. 逐token生成循环
    # 3. EOS停止检测
    # 4. 返回完整生成文本
```

**特性：**
- ✅ 完整的自回归生成循环
- ✅ 支持chat模板格式化
- ✅ EOS token停止检测
- ✅ 生成进度显示（每10个token）
- ✅ 正确的token累积和解码
- ✅ 保留原有`generate_next_token()`用于测试

### 2. `qwen3_webui_acl.py` - WebUI修复

#### 更新内容：
```python
# 旧代码（只生成单token）
next_token, logits, inference_time = engine.generate_next_token(...)
response = f"Next token: {next_token}"

# 新代码（完整文本生成）
response_text, inference_time = engine.generate_text(
    prompt,
    max_new_tokens=max_new_tokens,
    temperature=temperature
)
```

#### UI改进：
- 添加 `max_new_tokens` 滑块（32-256，默认128）
- 添加 `temperature` 滑块（0.1-2.0，默认1.0）
- 更新页脚说明文字

## 部署步骤

### 1. 同步文件到远程服务器

```bash
# 方法1: 使用提供的脚本
cd /home/oasis/Documents/ICT
./sync_qwen_fix.sh

# 方法2: 手动scp
scp remote_sync/qwen3_acl_inference.py HwHiAiUser@ict.local:/home/HwHiAiUser/ICT/
scp remote_sync/qwen3_webui_acl.py HwHiAiUser@ict.local:/home/HwHiAiUser/ICT/
```

### 2. 在远程服务器上重启服务

```bash
ssh HwHiAiUser@ict.local

# 停止现有服务
pkill -f qwen3_webui_acl.py

# 或者找到进程号后kill
ps aux | grep qwen3_webui
kill <PID>

# 重启WebUI
cd /home/HwHiAiUser/ICT
streamlit run qwen3_webui_acl.py --server.port 8502
```

### 3. 测试验证

访问 WebUI 并测试：
```
http://<远程服务器IP>:8502
```

测试对话：
- **输入**: "你好，请介绍一下你自己"
- **预期**: 应该返回完整的中文回复，不再是单个token或乱码

## 性能说明

### 当前实现特点：
- ✅ **可用性** - 能够生成完整对话
- ✅ **准确性** - 正确的token解码，无乱码
- ⚠️ **性能** - 每个token需要单独推理（无KV cache）

### 性能估算：
- 单token推理时间：~0.1-0.2s
- 生成128个token：~13-26秒
- 生成50个token：~5-10秒

### 推荐设置：
- 快速响应：`max_new_tokens=50`
- 正常对话：`max_new_tokens=128`
- 长回复：`max_new_tokens=256`（较慢）

## 技术细节

### 为什么没有乱码了？

**原代码问题：**
```python
# 只解码单个token
next_token_text = self.tokenizer.decode([next_token_id])
# 输出: "�" 或 "��" (不完整的UTF-8)
```

**修复后：**
```python
# 累积所有生成的token再解码
generated_ids = [token1, token2, token3, ...]
response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
# 输出: "我是一个AI助手" (完整文本)
```

### 停止token检测

```python
# Qwen2/Qwen3 特殊token
EOS_TOKEN_ID = 151643  # <|im_end|>
ENDOFTEXT = 151645     # <|endoftext|>

if next_token_id in [eos_token_id, 151643, 151645]:
    break
```

## 未来优化方向

1. **KV Cache** - 缓存attention状态，加速生成（需要模型导出支持）
2. **Batch推理** - 并行处理多个请求
3. **流式输出** - 逐token显示，提升用户体验
4. **量化优化** - INT8量化减少内存和提升速度

## 故障排查

### 如果仍然乱码：
1. 检查tokenizer路径是否正确
2. 确认模型是Qwen3-1.7B而非其他版本
3. 查看服务器端日志输出

### 如果生成太慢：
1. 减小`max_new_tokens`到50或更少
2. 检查NPU是否正常工作（`npu-smi info`）
3. 确认没有其他进程占用NPU

### 如果生成停不下来：
1. EOS token可能未正确检测
2. 检查tokenizer是否正确加载
3. 手动添加最大token限制

## 相关文件

- `remote_sync/qwen3_acl_inference.py` - 修复后的推理引擎
- `remote_sync/qwen3_webui_acl.py` - 修复后的WebUI
- `sync_qwen_fix.sh` - 一键同步脚本
- `QWEN3_FIX_README.md` - 本说明文档

## 修复时间

- 分析问题：2025-12-07
- 代码修复：2025-12-07
- 文档编写：2025-12-07
