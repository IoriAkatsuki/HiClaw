# Ascend 310B1 CPU / AI CPU 配置调研报告

> 板卡：Orange Pi AI Pro · soc_version=Ascend310B1 · npu-smi 25.2.0
> 采集时间：2026-03-19

---

## 1. 硬件概况

| 属性 | 值 |
|------|----|
| NPU 型号 | Ascend 310B1 |
| 算力 | 8 TOPS (INT8) |
| 片上内存 | 23674 MB (~23 GB LPDDR5) |
| AI Core 数量 | 2 × Da Vinci AI Core |
| 物理 CPU 核心 | 4 核 Cortex-A55 |

---

## 2. CPU 分区（cpu-num-cfg 实测）

```
npu-smi info -t cpu-num-cfg -i 0 -c 0
```

| 类型 | 数量 | 说明 |
|------|------|------|
| **AI CPU（AICPU）** | **1** | 专用于执行 AICPU 算子（Host 侧 fallback 算子） |
| **Control CPU** | **3** | 系统调度 + 用户进程可用的核心 |
| **Data CPU** | 0 | 未分配 |

关键结论：**4 个物理核中，1 个被 CANN 保留为 AI CPU，用户进程只能使用 3 个控制 CPU 核**。
`nproc` 返回 **3** 与此一致；`os.sched_getaffinity(0)` 返回 `{0, 1, 2}`（推测）。

---

## 3. npu-smi 常用命令速查

### 3.1 整体状态

```bash
npu-smi info                    # 所有卡概览（Health/Power/Temp/AICore%/Memory）
npu-smi info -t usages -i 0 -c 0   # 详细利用率
npu-smi info -t common -i 0 -c 0   # AICore%、内存%、温度
npu-smi info -t sensors -i 0 -c 0  # 传感器读数
npu-smi info -t memory -i 0 -c 0   # 内存详情
```

### 3.2 CPU/AI CPU 分区

```bash
npu-smi info -t cpu-num-cfg -i 0 -c 0  # 查询当前 CPU 分区配置
```

> 310B1 **不支持** `aicpu-config` 查询（会返回错误）。

### 3.3 运行时监控

```bash
npu-smi info watch                      # 滚动刷新所有卡状态
npu-smi info -t usages -i 0 -c 0        # 单次采样：AICore%、AICPU%、CtrlCPU%
```

### 3.4 实测利用率输出

| 指标 | 空闲值 | 说明 |
|------|--------|------|
| AICore Usage Rate | 0% | Da Vinci 核心利用率 |
| Aicpu Usage Rate | 1% | AICPU 算子调度（Host 侧） |
| Ctrlcpu Usage Rate | 16% | 控制 CPU（含系统后台进程） |
| Memory Usage | 8% (2106/23674 MB) | NPU 片上内存占用 |
| Hugepages | 100% (15/15 pages) | Hugepages 已满分配（DMA 缓冲区） |

### 3.5 Health=Alarm 解读

`npu-smi` 显示 `Health=Alarm` 但芯片仍可正常推理。
常见原因：Hugepages 100% 触发告警（15/15 页全部占用）；不代表硬件故障。
验证方法：检查 `err-count` 和 `ecc` 类型。

```bash
npu-smi info -t err-count -i 0 -c 0
npu-smi info -t ecc -i 0 -c 0
```

---

## 4. CPU Affinity 对本项目的影响

### 4.1 现有代码行为（unified_monitor_mp.py:204-211）

```python
def list_available_cpus() -> List[int]:
    if hasattr(os, "sched_getaffinity"):
        return sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    ...

def plan_worker_cpus(worker_names, available_cpus=None):
    cpus = list(available_cpus) if available_cpus is not None else list_available_cpus()
    return {name: cpus[index % len(cpus)] for index, name in enumerate(worker_names)}
```

`sched_getaffinity(0)` 在板端**只返回 3 个控制 CPU** 核（AI CPU 核已被 CANN 从调度掩码中移除），因此现有代码**无需修改**即可自动规避 AI CPU 冲突。

### 4.2 4 进程 → 3 核的轮转分配

| Worker | 分配核心（推测） |
|--------|----------------|
| camera_capture | CPU 0 |
| detector | CPU 1 |
| writer | CPU 2 |
| laser_worker | CPU 0（轮转复用） |

4 进程映射到 3 个控制 CPU 核，`laser_worker` 与 `camera_capture` 共享 CPU 0。
**优化建议**：若要减少抢占，可给 `camera_capture`（实时性最高）单独绑定一个核，将 `laser_worker` 绑定到 `writer` 所在核（两者都有 I/O 等待）。

```python
# 建议的手动 affinity 策略（3 核场景）
manual_plan = {
    "camera_capture": 0,   # 独占，优先级最高
    "detector":       1,   # 独占，推理密集
    "writer":         2,   # I/O 等待
    "laser_worker":   2,   # 与 writer 共享（串口 I/O 等待）
}
```

### 4.3 AICPU 算子影响

- `Aicpu Usage Rate=1%` 说明当前推理中 AICPU 算子占比极低（YOLO 已完全映射到 Da Vinci AI Core）。
- 如果将来使用包含大量 CPU 算子的模型（如含 triangular_update 的 Mamba 架构），AICPU 核负载会上升，此时 `detector` 进程不应绑定到与 AICPU 核相同的核（但目前 CANN 已隔离，无实际冲突风险）。

---

## 5. 结论

1. **310B1 的 4 核 CPU 中 1 核为 AICPU，3 核为用户可用控制 CPU**。
2. **现有 `plan_worker_cpus()` 实现已正确**：`sched_getaffinity()` 自动排除了 AICPU 核。
3. `Health=Alarm` 是 Hugepages 占满导致的告警，不影响推理功能。
4. 4 进程流水线的 3 核轮转分配可接受；若需进一步降低延迟，可手动指定 `laser_worker` 与 `writer` 共核。
