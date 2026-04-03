# Qwen3-1.7B 修复验证指南

## 部署状态

✅ **已完成:**
1. 修复了推理引擎 - 添加完整文本生成功能
2. 修复了WebUI - 使用新的生成方法
3. 修正了模型路径配置
4. 部署到远程服务器 (10.42.0.190)
5. WebUI服务已启动在端口8502

## 访问WebUI

**URL**: http://10.42.0.190:8502

## 测试步骤

### 测试1: 中文对话（验证乱码已修复）

1. 访问 WebUI
2. 输入: **"你好，请介绍一下你自己"**
3. **预期结果**:
   - 应返回完整的中文句子（不是单个token）
   - 没有乱码字符
   - 回复长度至少20-50个字符

### 测试2: 英文对话

1. 输入: **"What is artificial intelligence?"**
2. **预期结果**:
   - 完整的英文句子
   - 语法正确
   - 有实质内容

### 测试3: 参数调整

1. 在侧边栏调整参数:
   - `Max new tokens`: 50 → 100
   - `Temperature`: 1.0 → 0.7
2. 输入同样的问题
3. 观察回复长度和风格变化

## 性能基准

**预期性能:**
- 首次加载模型: 5-10秒
- 生成50 tokens: 5-10秒
- 生成100 tokens: 10-20秒
- 每token推理: ~0.1-0.2秒

**NPU状态:**
```bash
ssh HwHiAiUser@10.42.0.190 "npu-smi info"
```
预期内存使用: 5-7GB

## 修复对比

### 修复前（有问题）
```
用户: "你好"
模型: "�" 或 "吗" (单个乱码token)
```

### 修复后（预期）
```
用户: "你好"
模型: "你好！我是一个AI助手，很高兴为你服务。请问有什么我可以帮助你的吗？"
```

## 故障排查

### 如果WebUI无法访问

```bash
# 检查进程
ssh HwHiAiUser@10.42.0.190 "ps aux | grep streamlit"

# 重启服务
cd /home/oasis/Documents/ICT
./start_qwen3_webui_remote.sh

# 查看日志
ssh HwHiAiUser@10.42.0.190 "tail -f /home/HwHiAiUser/ICT/qwen3_webui.log"
```

### 如果仍然出现乱码

1. 检查WebUI使用的是否是新的 `generate_text()` 方法
2. 查看服务器日志中的生成过程
3. 验证模型文件: `qwen3_1_7b_fp16.om`
4. 检查tokenizer路径: `models/qwen3-1.7b`

### 如果生成速度太慢

1. 减少 `max_new_tokens` 到 50 或更少
2. 检查NPU温度（不应超过85°C）
3. 确认没有其他进程占用NPU

## 服务管理命令

```bash
# 启动WebUI
./start_qwen3_webui_remote.sh

# 停止WebUI
ssh HwHiAiUser@10.42.0.190 "pkill -f qwen3_webui_acl.py"

# 查看实时日志
ssh HwHiAiUser@10.42.0.190 "tail -f /home/HwHiAiUser/ICT/qwen3_webui.log"

# 查看NPU状态
ssh HwHiAiUser@10.42.0.190 "npu-smi info"

# 同步最新代码
cd /home/oasis/Documents/ICT
./sync_qwen_fix.sh
```

## 技术细节

### 关键修改

**qwen3_acl_inference.py:205-280**
- 新增 `generate_text()` 方法
- 实现完整的token累积循环
- 添加EOS停止检测
- 支持chat模板格式化

**qwen3_webui_acl.py:93-102**
- 使用 `generate_text()` 替换 `generate_next_token()`
- 添加temperature和max_new_tokens参数控制

### 配置文件

**模型**: `/home/HwHiAiUser/ICT/qwen3_1_7b_fp16.om`
**Tokenizer**: `/home/HwHiAiUser/ICT/models/qwen3-1.7b`
**端口**: 8502
**主机**: 10.42.0.190

## 下一步优化建议

1. **KV Cache** - 缓存attention状态，加速生成5-10倍
2. **流式输出** - 实时显示生成的token，提升体验
3. **批量推理** - 支持多用户并发
4. **模型量化** - INT8量化减少内存占用

## 修复时间线

- 2025-12-07 13:00 - 问题诊断
- 2025-12-07 14:00 - 代码修复完成
- 2025-12-07 18:00 - 部署到远程服务器
- 2025-12-07 18:05 - WebUI启动成功
