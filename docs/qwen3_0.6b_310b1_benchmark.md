# Qwen3-0.6B Ascend 310B1 NPU Benchmark

## Model

| Key | Value |
|-----|-------|
| Model | Qwen/Qwen3-0.6B |
| Parameters | 0.6B |
| Architecture | Standard GQA Transformer (28 layers) |
| Hidden size | 1024 |
| Attention heads | 16 Q / 8 KV |
| Vocab size | 151,936 |
| Precision | FP16 (allow_fp32_to_fp16) |

## Pipeline

```
PyTorch (safetensors)
  → torch.onnx.export (legacy, opset 17, use_cache=False)
  → 29 standard ONNX ops, 4106 nodes (full model)
  → ATC cross-compile (x86 → Ascend310B1)
  → .om (1.5 GB)
  → ACL inference on Orange Pi AI Pro
```

## Hardware

| Key | Value |
|-----|-------|
| Board | Orange Pi AI Pro |
| SoC | Ascend 310B1 |
| AI Core | 1 |
| NPU Memory | ~6 GB |
| CANN | 8.5.0 |

## WikiText-2 Perplexity

| Metric | Value |
|--------|-------|
| **PPL** | **22.44** |
| Tokens evaluated | 299,077 |
| Chunks | 1,168 |
| Sequence length | 512 |
| Stride | 256 |
| Total time | 4,579 s (76.3 min) |
| **Throughput (prefill)** | **65.3 tok/s** |

## Correctness Verification

Prompt `1+1=`, top-5 next-token predictions:

| Rank | Token | PyTorch CPU | NPU .om | Delta |
|------|-------|-------------|---------|-------|
| 1 | `'2'` | 17.578 | 17.578 | 0.000 |
| 2 | `'1'` | 15.406 | 15.398 | 0.008 |
| 3 | `'?\n\n'` | 14.258 | 14.258 | 0.000 |
| 4 | `'?\n'` | 14.055 | 14.055 | 0.000 |
| 5 | `'3'` | 13.594 | 13.594 | 0.000 |

Rankings identical. Max score delta < 0.01 (FP16 precision).

## CPU vs NPU Inference Comparison

| Mode | Quant | Prefill | Decode | KV Cache | Chat Usable |
|------|-------|---------|--------|----------|-------------|
| NPU .om (prefill-only) | FP16 | 65.3 tok/s | 0.3 tok/s* | No | Slow |
| **CPU llama.cpp (3 threads)** | **Q4_K_M** | **24.4 tok/s** | **7.4 tok/s** | **Yes** | **Yes** |

*NPU decode is 0.3 tok/s because each step re-runs full 512-token prefill (no KV cache).

**Conclusion:** CPU llama.cpp with Q4 quantization is **24x faster for generation** than NPU prefill-only,
because llama.cpp has native KV cache support. NPU prefill is 2.7x faster but cannot do efficient decode
due to 310B1's lack of dynamic middle-dimension shape support in .om format.

**Recommended deployment:** NPU for YOLO detection, CPU for LLM chat (with DVPP offloading resize+JPEG
to free ~1.1 CPU cores for llama.cpp).

## DVPP Hardware Offloading

| Operation | cv2 (CPU) | DVPP (hardware) | CPU freed |
|-----------|-----------|-----------------|-----------|
| Resize 640→640 | 1.6ms | 4.2ms | ~0.3 core |
| JPEG encode | 5.3ms | 3.5ms | ~0.8 core |
| **Total** | | | **~1.1 cores** |

DVPP latency slightly higher (H2D/D2H copy overhead) but CPU utilization drops to zero for these ops.

## KV Cache on 310B1: Why It Failed

| Attempt | Result |
|---------|--------|
| TorchScript trace (no dynamic_axes) | Shapes baked as constants |
| TorchScript trace + dynamic_axes | ONNX shapes symbolic ✓ |
| ATC gear compilation | `Multi-batch not support middle dynamic shape` ❌ |
| ATC --input_shape_range | Runtime error 500002 (unsupported) ❌ |
| torch.export (dynamo) | Can't handle DynamicCache ❌ |

Root cause: Ascend 310B1 ATC does not support dynamic shapes on middle dimensions (dim=2 of KV cache `[1,8,seq,128]`). This is a hardware/compiler limitation, not a graph issue.

## Failed Alternatives

| Path | Failure |
|------|---------|
| llama.cpp ggml-cann | `aclopExecuteV2("MatMul")` MatchOpModel fail |
| torch_npu | `FlashAttentionScore` not implemented on 310B1 |
| Qwen3.5-0.8B (hybrid arch) | linear_attention 5D tensors trigger TBE codegen bug |
| onnx-community pre-exported | com.microsoft contrib ops (GQA, RotaryEmbedding) |

## Date

2026-03-21
