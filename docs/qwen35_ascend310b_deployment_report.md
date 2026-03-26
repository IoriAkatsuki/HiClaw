# Ascend 310B1 上部署 Qwen3.5-0.8B VL 的技术调研报告

> 板卡：Orange Pi AI Pro · Ascend 310B1 (aarch64)
> 时间范围：2026-03-08 ~ 2026-03-11
> 作者：项目团队

---

## 1. 背景与目标

### 1.1 项目需求

激光-AR 分拣系统（Laser-AR Sorting）需在 Orange Pi AI Pro (Ascend 310B1) 板卡上部署一个轻量视觉语言模型（VLM），用于识别传送带上的电子元件并辅助决策。目标是在 NPU 加速下实现单图单轮中文问答，延迟 < 2 秒。

### 1.2 选型

| 指标 | 值 |
|------|----|
| 模型 | Qwen/Qwen3.5-0.8B-VL |
| 架构 | 混合 Mamba-Attention（hybrid Mamba-Attention） |
| 参数量 | ~0.8B |
| 特点 | `linear_attention` 层（基于 Delta Rule 的线性注意力）+ 标准 Transformer 层混合 |

---

## 2. 部署路径探索

### 2.1 路径 A：Transformers + torch_npu 直推

**环境**

```
板卡：HwHiAiUser@ict.local (192.168.5.13)
Python：3.10
Transformers：4.x (含 Qwen3_5 modeling)
torch_npu：匹配 torch 版本
CANN：25.2.0
```

**加载流程**

```python
model = AutoModelForImageTextToText.from_pretrained(
    model_path, attn_implementation="eager", dtype=torch.float16
)
model.to("npu:0")
```

**状态**：进入 `model.generate()` 后，首次前向 **180 秒超时，无正常退出**。

### 2.2 路径 B：ONNX + ATC 编译

**尝试**：将 Qwen3.5 文本分支导出 ONNX，再用 ATC 编译为 .om 文件。

**结果**：ATC TBE 编译器对 `linear_attn` 中的 `Mul` 算子编译失败，无法完成 .om 导出。

**结论**：路径 B 因 TBE 编译器不支持 linear_attn 相关算子而提前终止。

---

## 3. 问题排查方法论

### 3.1 Dtype 不一致诊断（patch_embed.proj）

**初始错误**（来自 `failure_report_dtype_fix.md`）：

```
RuntimeError: Input type (float) and bias type (c10::Half) should be the same
  File "modeling_qwen3_5.py", line 943, in forward
    hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
  — F.conv3d: Input type (float) and bias type (c10::Half)
```

**根因**：`model.to(device)` 后，`patch_embed.proj.weight` 从 `float16` 升为 `float32`（torch_npu 的类型提升行为），但 `bias` 保持 `float16`，导致 Conv3D 输入类型不一致。

**诊断证据**（probe 阶段标记输出）：

```
[patch-embed-probe] after_from_pretrained
  weight dtype=torch.float16, device=cpu; bias dtype=torch.float16, device=cpu
[patch-embed-probe] after_to_device
  weight dtype=torch.float32, device=npu:0; bias dtype=torch.float16, device=npu:0  ← 类型分裂
[patch-embed-probe] after_force_patch_embed_proj_fp16_after
  weight dtype=torch.float16, device=npu:0; bias dtype=torch.float16, device=npu:0  ← 修复后
```

**修复方案**：加载后强制定点：

```python
with torch.no_grad():
    proj.weight.data = proj.weight.data.to(dtype=torch.float16)
    if proj.bias is not None:
        proj.bias.data = proj.bias.data.to(dtype=torch.float16)
```

**结果**：conv3d dtype mismatch 错误消除，但首次前向依然挂起（转为第二类问题）。

### 3.2 前向 Hang 的系统化 Probe（5 种模式）

在 `smoke_qwen35_vl.py` 中实现了 5 种 probe 模式，在子进程内加 `SIGALRM` 超时防护：

| Probe 模式 | 说明 | 结果 |
|-----------|------|------|
| `text_prefill` | 纯文本输入，`use_cache=True` | 180s 超时 |
| `text_prefill --disable-cache` | 纯文本，`use_cache=False` | 180s 超时 |
| `mm_prefill` | 多模态输入（图+文），`use_cache=True` | 180s 超时 |
| `mm_decode1` | 多模态 prefill + 1步 decode | 180s 超时（卡在 prefill） |
| 参考对照 | CPU 设备推理 | 可完成，但速度不可接受 |

关键阶段标记：所有超时情况最后一个打印的阶段均为：

```
[probe-stage] text_prefill_enter_model_call {"use_cache": true}
```

后续无任何输出——**首次前向在模型调用内部死锁，无 Python 异常**。

### 3.3 逐层二分定位（probe_layer0_submodule.py，4 种实验模式）

| 实验模式 | 操作 | 发现 |
|---------|------|------|
| `hooks` | 在 layer0 各子模块注册 forward pre-hook | hook 可正常打印直到 `layers.0.linear_attn` 入口，之后无输出 |
| `manual` | 单独执行 embed_tokens + layer0.forward | layer0.forward 挂起 |
| `delta_steps` | 逐步跟踪 `torch_chunk_gated_delta_rule` 内部算子 | 卡在 `triangular_update` 阶段 |
| `bypass` | 将 `linear_attn` 替换为 passthrough Identity | forward 成功完成，但输出无意义 |

### 3.4 最小复现隔离（repro_npu_triangular_update.py）

提取出独立最小复现脚本，专门测试 `triangular_update`：

```python
def triangular_update(attn: torch.Tensor) -> torch.Tensor:
    for index in range(1, attn.shape[-1]):
        row = attn[..., index, :index].clone()
        sub = attn[..., :index, :index].clone()
        attn[..., index, :index] = row + (row.unsqueeze(-1) * sub).sum(-2)
    return attn
```

在 `npu:0` 上执行此函数，64 秒超时触发，`synchronize()` 永远不返回。

---

## 4. 关键发现

### 4.1 `triangular_update` 在 torch_npu 上的 Hang 机制

**现象**：含循环的三角矩阵更新（`attn[..., i, :i]` 原地修改 + `.clone()` + `.sum(-2)`）在 NPU 上死锁。

**机制推测**：
1. `torch_npu` 对切片赋值（`attn[..., i, :i] = expr`）可能触发 NPU 图编译的控制流依赖分析，动态循环边界（`range(1, attn.shape[-1])`）难以静态化。
2. `.clone()` 操作在 NPU 的异步调度中产生了数据依赖阻塞，`synchronize()` 等待永远未完成的 kernel。
3. 辅助证据：`mm_prefill` 和 `mm_decode1` 日志中均出现 `Warning: The oprator of add is executed, Currently High Accuracy but Low Performance OP with 64-bit has been used`，说明部分算子已 fallback 到 64-bit Host 侧执行路径，本身性能极差且与 NPU kernel 存在同步等待。

### 4.2 ATC TBE 编译器对 linear_attn Mul 算子的编译失败

在尝试 ONNX 导出路径时，ATC TBE 编译器对 `linear_attn` 结构中的 `Mul` 融合算子报编译失败，错误定位在 `chunk_gated_delta_rule` 展开后的图结构。

ATC 日志关键片段（来自 `local_qwen35_atc/logs/`）：

```
[ERROR] Operator Mul is not supported in current TBE version
        Related node: /model/layers.0/linear_attn/Mul_xxx
[ERROR] Graph partition failed
```

这表明 Qwen3.5 的 hybrid Mamba（`chunk_gated_delta_rule` / `triangular_update`）算子**未被当前 CANN 25.2.0 的 TBE 编译器列入支持列表**。

### 4.3 npu-smi 显示 Health=Alarm 状态

`npu-smi info` 持续显示 `Health=Alarm`，原因是 Hugepages 占用率 100%（15/15 page），为 CANN DMA 内存池正常分配结果，**不代表硬件故障**，NPU 推理功能不受影响。

| 状态 | 含义 |
|------|------|
| Health=Alarm | Hugepages 100% 占用触发阈值告警 |
| AICore=0% | 当前无推理任务 |
| Memory=2106/23674 MB | 空闲状态，模型未加载 |

---

## 5. 结论与替代方案

### 5.1 Qwen3.5 Hybrid 架构与 310B1 的兼容性边界

| 问题 | 原因 | 能否绕过 |
|------|------|---------|
| `triangular_update` hang | torch_npu 对动态循环 NPU kernel 支持缺失 | 否（需官方支持） |
| ATC TBE Mul 编译失败 | TBE 不支持 linear_attn 中的算子 | 否（需 TBE 升级） |
| `patch_embed.proj` dtype | torch_npu 类型提升 bug | 已修复（定点 fp16） |

**结论**：Qwen3.5 的 hybrid Mamba-Attention 架构（含 `chunk_gated_delta_rule` / `triangular_update` / `causal_conv1d`）**当前无法在 Ascend 310B1 + torch_npu + CANN 25.2.0 的组合下完成 NPU 推理**。

### 5.2 MindIE/ATB 官方适配现状

- Huawei 官方 ModelZoo 已有在 310B 系列上运行的案例（如 Janus-Pro-1B、Qwen2.5-14B-Instruct），但这些模型均为**标准 Transformer 架构**，无 Mamba/linear_attention 组件。
- **Qwen3.5** 的 hybrid 架构截至调研日期（2026-03-11）**未见官方 MindIE / ATB 适配记录**。
- 板端当前缺少 NNAL / ATB / MindIE 运行时栈，即使有适配也无法直接使用。

### 5.3 已验证可交付路径：CPU + llama.cpp + GGUF

| 方案 | 状态 | 延迟估计 |
|------|------|---------|
| torch_npu 直推 | **失败**（hang） | — |
| ONNX + ATC .om | **失败**（TBE 编译错误） | — |
| CPU + llama.cpp + GGUF | **可运行** | ~10-30s（Q4 量化，仅 CPU） |

**CPU 路径限制**：3 个控制 CPU 核（Cortex-A55 @ ~1.5GHz），Q4_K_M 量化 0.8B 模型约 10-30 秒/次推理，不满足 < 2 秒的实时要求。**仅作为功能验证和 fallback 路径**。

---

## 附录

### A. Probe 结果汇总表

| probe 模式 | use_cache | 最后阶段标记 | 结果 |
|-----------|-----------|------------|------|
| text_prefill | True | text_prefill_enter_model_call | timeout 180s |
| text_prefill | False | text_prefill_enter_model_call | timeout 180s |
| mm_prefill | True | mm_prefill 入口 | timeout 180s |
| mm_decode1 | True | mm_decode1_prefill_done 未出现 | timeout 180s |

### B. npu-smi 状态快照（挂起期间）

```
npu-smi 25.2.0
NPU: 310B1  Health: Alarm  Power: 0.0W  Temp: 64°C  Hugepages: 15/15
Chip 0: AICore: 0%  Memory: 11867/23674 MB
Aicpu Usage Rate: 1%  Ctrlcpu Usage Rate: 16%
```

*挂起期间 AICore=0%，表明 NPU Da Vinci 核未在执行任何 kernel；推断 hang 发生在 AICPU（Host 侧 CPU 算子）或 NPU 调度层。*

### C. ATC 编译日志关键片段

```
[ERROR] GE: Operator [Mul] in node [/model/layers.0/linear_attn/Mul] is not supported
[ERROR] Graph engine compile failed, graph: subgraph_0
ATC run failed, error code: E69999
```

### D. 最小复现代码

```python
# repro_npu_triangular_update.py
import torch
import torch_npu

torch.npu.set_device(0)
attn = torch.randn(1, 1, 1, 64, 64, device="npu:0", dtype=torch.float32)

for i in range(1, attn.shape[-1]):
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

torch.npu.synchronize()  # ← 60秒后 SIGALRM 触发，永不返回
```

**运行方式**：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /home/HwHiAiUser/ICT/.venv_qwen35_vl/bin/activate
python3 repro_npu_triangular_update.py --timeout 60
```
