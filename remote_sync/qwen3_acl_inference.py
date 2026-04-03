#!/usr/bin/env python3
"""
Qwen ACL 推理引擎（OrangePi AI Pro / Ascend 310B）

优化点：
1. 复用 ACL dataset/data_buffer，减少每 token 的创建销毁开销。
2. 复用输入工作区和输出 host 缓冲，降低内存抖动。
3. 采样参数（temperature/top_k/top_p/repetition_penalty）真正生效。
4. 提供流式生成与基准测试能力，输出可观测指标。
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from typing import Dict, Generator, List, Optional, Sequence

import acl
import numpy as np
from transformers import AutoTokenizer

# ACL Constants (for compatibility across versions)
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


@dataclass
class GenerationConfig:
    """生成参数。"""

    max_new_tokens: int = 128
    temperature: float = 0.9
    top_k: int = 40
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.0
    stop_token_ids: Optional[Sequence[int]] = None


class Qwen3ACLInference:
    def __init__(self, model_path, tokenizer_path, device_id=0, max_seq_len=512):
        self.device_id = device_id
        self.model_path = model_path
        self.max_seq_len = max_seq_len

        print(f"[Init] Loading tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.vocab_size = len(self.tokenizer)

        # ACL resources
        self.context = None
        self.stream = None
        self.model_id = None
        self.model_desc = None

        # Model info
        self.input_num = 0
        self.output_num = 0
        self.input_sizes: List[int] = []
        self.output_sizes: List[int] = []

        # Pre-allocated device buffers
        self.input_buffer_ptrs: List[int] = []
        self.output_buffer_ptrs: List[int] = []

        # Reusable dataset/data_buffer
        self.input_dataset = None
        self.output_dataset = None
        self.input_data_buffers: List[int] = []
        self.output_data_buffers: List[int] = []

        # Reusable host output buffers
        self.output_host_arrays: List[np.ndarray] = []
        self.output_host_ptrs: List[int] = []

        # Reusable input working buffers
        self.input_ids_buffer = np.zeros((1, self.max_seq_len), dtype=np.int64)
        self.attention_mask_buffer = np.zeros((1, self.max_seq_len), dtype=np.int64)
        self.position_ids_buffer = np.arange(self.max_seq_len, dtype=np.int64).reshape(1, -1)

        # Runtime stats
        self.last_generation_stats: Dict[str, float] = {}

        # Initialize ACL
        self.init_acl()

    def check_ret(self, ret, message):
        """检查 ACL 返回码。"""
        if ret != 0:
            raise RuntimeError(f"{message} failed with ret={ret}")

    def init_acl(self):
        """初始化 ACL 并加载模型。"""
        print("[ACL] Initializing ACL runtime...")
        ret = acl.init()
        self.check_ret(ret, "acl.init")

        ret = acl.rt.set_device(self.device_id)
        self.check_ret(ret, "acl.rt.set_device")

        self.context, ret = acl.rt.create_context(self.device_id)
        self.check_ret(ret, "acl.rt.create_context")

        self.stream, ret = acl.rt.create_stream()
        self.check_ret(ret, "acl.rt.create_stream")

        print(f"[ACL] Loading model: {self.model_path}")
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        self.check_ret(ret, "acl.mdl.load_from_file")

        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        self.check_ret(ret, "acl.mdl.get_desc")

        self.input_num = acl.mdl.get_num_inputs(self.model_desc)
        self.output_num = acl.mdl.get_num_outputs(self.model_desc)

        print("[ACL] ✓ Model loaded successfully")
        print(f"      - Inputs: {self.input_num}")
        print(f"      - Outputs: {self.output_num}")

        self._cache_io_sizes()
        self._allocate_buffers()
        self._build_reusable_datasets()
        self._prepare_host_output_buffers()

    def _cache_io_sizes(self):
        self.input_sizes = [
            acl.mdl.get_input_size_by_index(self.model_desc, i) for i in range(self.input_num)
        ]
        self.output_sizes = [
            acl.mdl.get_output_size_by_index(self.model_desc, i) for i in range(self.output_num)
        ]

        for i, size in enumerate(self.input_sizes):
            print(f"      - Input[{i}] size: {size} bytes")
        for i, size in enumerate(self.output_sizes):
            print(f"      - Output[{i}] size: {size} bytes")

    def _allocate_buffers(self):
        """预分配 device buffer。"""
        print("[ACL] Pre-allocating input/output device buffers...")
        for i, size in enumerate(self.input_sizes):
            device_ptr, ret = acl.rt.malloc(size, 0)
            self.check_ret(ret, f"acl.rt.malloc input[{i}]")
            self.input_buffer_ptrs.append(device_ptr)

        for i, size in enumerate(self.output_sizes):
            device_ptr, ret = acl.rt.malloc(size, 0)
            self.check_ret(ret, f"acl.rt.malloc output[{i}]")
            self.output_buffer_ptrs.append(device_ptr)

    def _build_reusable_datasets(self):
        """创建可复用 dataset/data_buffer。"""
        self.input_dataset = acl.mdl.create_dataset()
        for i, (ptr, size) in enumerate(zip(self.input_buffer_ptrs, self.input_sizes)):
            data_buffer = acl.create_data_buffer(ptr, size)
            self.input_data_buffers.append(data_buffer)
            ret = acl.mdl.add_dataset_buffer(self.input_dataset, data_buffer)
            self.check_ret(ret, f"acl.mdl.add_dataset_buffer input[{i}]")

        self.output_dataset = acl.mdl.create_dataset()
        for i, (ptr, size) in enumerate(zip(self.output_buffer_ptrs, self.output_sizes)):
            data_buffer = acl.create_data_buffer(ptr, size)
            self.output_data_buffers.append(data_buffer)
            ret = acl.mdl.add_dataset_buffer(self.output_dataset, data_buffer)
            self.check_ret(ret, f"acl.mdl.add_dataset_buffer output[{i}]")

    def _prepare_host_output_buffers(self):
        """预分配 host 输出缓存（numpy），避免每次推理分配。"""
        for size in self.output_sizes:
            # 常见 logits 输出为 float32/float16，这里按字节对齐做通用处理。
            if size % 4 == 0:
                arr = np.empty(size // 4, dtype=np.float32)
            else:
                arr = np.empty(size // 2, dtype=np.float16)
            self.output_host_arrays.append(arr)
            self.output_host_ptrs.append(acl.util.numpy_to_ptr(arr))

    def _normalize_generation_config(
        self,
        config: Optional[GenerationConfig] = None,
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
    ) -> GenerationConfig:
        if config is not None:
            cfg = GenerationConfig(
                max_new_tokens=int(config.max_new_tokens),
                temperature=float(config.temperature),
                top_k=int(config.top_k),
                top_p=float(config.top_p),
                do_sample=bool(config.do_sample),
                repetition_penalty=float(config.repetition_penalty),
                stop_token_ids=list(config.stop_token_ids) if config.stop_token_ids is not None else None,
            )
        else:
            cfg = GenerationConfig()

        if max_new_tokens is not None:
            cfg.max_new_tokens = int(max_new_tokens)
        if temperature is not None:
            cfg.temperature = float(temperature)
        if top_k is not None:
            cfg.top_k = int(top_k)
        if top_p is not None:
            cfg.top_p = float(top_p)
        if do_sample is not None:
            cfg.do_sample = bool(do_sample)
        if repetition_penalty is not None:
            cfg.repetition_penalty = float(repetition_penalty)
        if stop_token_ids is not None:
            cfg.stop_token_ids = list(stop_token_ids)

        if cfg.stop_token_ids is None:
            stop_ids = [151643, 151645]  # Qwen 常见停止 token
            if self.tokenizer.eos_token_id is not None:
                stop_ids.append(int(self.tokenizer.eos_token_id))
            cfg.stop_token_ids = sorted(set(stop_ids))

        cfg.max_new_tokens = max(1, int(cfg.max_new_tokens))
        cfg.top_k = max(0, int(cfg.top_k))
        cfg.top_p = min(1.0, max(0.0, float(cfg.top_p)))
        cfg.temperature = float(cfg.temperature)
        cfg.repetition_penalty = max(1.0, float(cfg.repetition_penalty))
        return cfg

    def _prepare_step_inputs(self, generated_ids: Sequence[int]) -> int:
        """把当前上下文写入复用输入缓冲区，返回有效 seq_len。"""
        current_ids = generated_ids[-self.max_seq_len :]
        seq_len = len(current_ids)

        self.input_ids_buffer.fill(0)
        self.attention_mask_buffer.fill(0)
        self.input_ids_buffer[0, :seq_len] = current_ids
        self.attention_mask_buffer[0, :seq_len] = 1
        return seq_len

    def inference(self, input_ids, attention_mask, position_ids):
        """
        运行单次模型推理。
        返回: (outputs, timing_breakdown)
        """
        ret = acl.rt.set_context(self.context)
        self.check_ret(ret, "acl.rt.set_context")

        candidate_inputs = [input_ids, attention_mask, position_ids]
        if self.input_num > len(candidate_inputs):
            raise RuntimeError(f"unsupported input_num={self.input_num}, expected <= {len(candidate_inputs)}")
        inputs = candidate_inputs[: self.input_num]

        t_h2d = 0.0
        t_exec = 0.0
        t_d2h = 0.0

        h2d_start = time.perf_counter()
        for idx, input_data in enumerate(inputs):
            if input_data.dtype != np.int64:
                input_data = input_data.astype(np.int64, copy=False)
            if not input_data.flags["C_CONTIGUOUS"]:
                input_data = np.ascontiguousarray(input_data)

            expected_size = self.input_sizes[idx]
            data_size = input_data.nbytes
            if data_size != expected_size:
                raise ValueError(
                    f"input[{idx}] size mismatch: got={data_size}, expected={expected_size}"
                )

            host_ptr = acl.util.numpy_to_ptr(input_data)
            ret = acl.rt.memcpy(
                self.input_buffer_ptrs[idx],
                expected_size,
                host_ptr,
                data_size,
                ACL_MEMCPY_HOST_TO_DEVICE,
            )
            self.check_ret(ret, f"acl.rt.memcpy input[{idx}]")
        t_h2d = (time.perf_counter() - h2d_start) * 1000.0

        exec_start = time.perf_counter()
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        self.check_ret(ret, "acl.mdl.execute")
        t_exec = (time.perf_counter() - exec_start) * 1000.0

        d2h_start = time.perf_counter()
        outputs: List[np.ndarray] = []
        for i, size in enumerate(self.output_sizes):
            ret = acl.rt.memcpy(
                self.output_host_ptrs[i],
                size,
                self.output_buffer_ptrs[i],
                size,
                ACL_MEMCPY_DEVICE_TO_HOST,
            )
            self.check_ret(ret, f"acl.rt.memcpy output[{i}]")
            outputs.append(self.output_host_arrays[i])
        t_d2h = (time.perf_counter() - d2h_start) * 1000.0

        return outputs, {"h2d_ms": t_h2d, "execute_ms": t_exec, "d2h_ms": t_d2h}

    def _extract_last_logits(
        self, logits_flat: np.ndarray, seq_len: int, attention_mask: np.ndarray
    ) -> np.ndarray:
        """
        从输出张量中提取最后一个有效 token 的 logits。
        """
        flat = logits_flat.astype(np.float32, copy=False)
        valid_len = int(np.sum(attention_mask[0]))
        valid_len = max(1, valid_len)
        last_pos = min(valid_len - 1, seq_len - 1)

        # 常见形态： [max_seq_len, vocab_aligned] 展平
        if flat.size % self.max_seq_len == 0:
            aligned_vocab = flat.size // self.max_seq_len
            reshaped = flat.reshape(self.max_seq_len, aligned_vocab)
            picked = reshaped[last_pos, : min(aligned_vocab, self.vocab_size)]
            if picked.size < self.vocab_size:
                out = np.full(self.vocab_size, -1e9, dtype=np.float32)
                out[: picked.size] = picked
                return out
            return picked.astype(np.float32, copy=False)

        # 兜底：直接取最后 vocab_size 个值
        if flat.size >= self.vocab_size:
            return flat[-self.vocab_size :].astype(np.float32, copy=False)

        out = np.full(self.vocab_size, -1e9, dtype=np.float32)
        out[: flat.size] = flat
        return out

    def _apply_repetition_penalty(
        self, logits: np.ndarray, generated_ids: Sequence[int], penalty: float
    ) -> np.ndarray:
        if penalty <= 1.0 or not generated_ids:
            return logits
        adjusted = logits.copy()
        for token_id in set(generated_ids[-256:]):
            if 0 <= token_id < adjusted.size:
                if adjusted[token_id] < 0:
                    adjusted[token_id] *= penalty
                else:
                    adjusted[token_id] /= penalty
        return adjusted

    @staticmethod
    def _stable_softmax(logits: np.ndarray) -> np.ndarray:
        finite_mask = np.isfinite(logits)
        if not np.any(finite_mask):
            probs = np.zeros_like(logits, dtype=np.float64)
            probs[np.argmax(logits)] = 1.0
            return probs

        safe_logits = logits.copy()
        safe_logits[~finite_mask] = -1e9
        max_logit = np.max(safe_logits)
        exp_vals = np.exp(safe_logits - max_logit)
        exp_vals[~finite_mask] = 0.0
        denom = np.sum(exp_vals)
        if denom <= 0:
            probs = np.zeros_like(safe_logits, dtype=np.float64)
            probs[np.argmax(safe_logits)] = 1.0
            return probs
        return exp_vals / denom

    def _apply_top_k_top_p(self, logits: np.ndarray, top_k: int, top_p: float) -> np.ndarray:
        out = logits.copy()

        if 0 < top_k < out.size:
            keep_idx = np.argpartition(out, -top_k)[-top_k:]
            mask = np.full_like(out, -np.inf)
            mask[keep_idx] = out[keep_idx]
            out = mask

        if 0.0 < top_p < 1.0:
            sorted_idx = np.argsort(out)[::-1]
            sorted_logits = out[sorted_idx]
            sorted_probs = self._stable_softmax(sorted_logits)
            cumsum = np.cumsum(sorted_probs)
            cutoff = int(np.searchsorted(cumsum, top_p, side="right")) + 1
            cutoff = max(1, cutoff)
            keep_sorted_idx = sorted_idx[:cutoff]
            mask = np.full_like(out, -np.inf)
            mask[keep_sorted_idx] = out[keep_sorted_idx]
            out = mask

        return out

    def _sample_next_token(
        self, last_logits: np.ndarray, generated_ids: Sequence[int], cfg: GenerationConfig
    ) -> int:
        logits = self._apply_repetition_penalty(last_logits, generated_ids, cfg.repetition_penalty)

        if not cfg.do_sample or cfg.temperature <= 0:
            return int(np.argmax(logits))

        filtered = self._apply_top_k_top_p(logits, cfg.top_k, cfg.top_p)
        scaled = filtered / max(cfg.temperature, 1e-6)
        probs = self._stable_softmax(scaled).astype(np.float64)

        try:
            next_token = int(np.random.choice(np.arange(probs.size), p=probs))
        except ValueError:
            next_token = int(np.argmax(filtered))
        return next_token

    def stream_generate(
        self,
        prompt: str,
        *,
        config: Optional[GenerationConfig] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        stop_token_ids: Optional[Sequence[int]] = None,
    ) -> Generator[Dict[str, object], None, None]:
        """
        流式生成：每次 yield 一个包含 delta/text/step 的字典。
        """
        cfg = self._normalize_generation_config(
            config,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            stop_token_ids=stop_token_ids,
        )
        stop_ids = set(cfg.stop_token_ids or [])

        total_start = time.perf_counter()
        tokenize_start = time.perf_counter()

        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted_prompt = prompt
        else:
            formatted_prompt = prompt

        prompt_ids = self.tokenizer.encode(formatted_prompt, add_special_tokens=True)
        tokenize_ms = (time.perf_counter() - tokenize_start) * 1000.0
        generated_ids = list(prompt_ids)

        emitted_text = ""
        step_metrics: List[Dict[str, float]] = []
        generated_token_count = 0

        for step in range(cfg.max_new_tokens):
            step_start = time.perf_counter()
            seq_len = self._prepare_step_inputs(generated_ids)

            outputs, infer_timing = self.inference(
                self.input_ids_buffer, self.attention_mask_buffer, self.position_ids_buffer
            )
            last_logits = self._extract_last_logits(outputs[0], seq_len, self.attention_mask_buffer)
            next_token_id = self._sample_next_token(last_logits, generated_ids, cfg)
            step_ms = (time.perf_counter() - step_start) * 1000.0

            step_metrics.append(
                {
                    "step_ms": step_ms,
                    "h2d_ms": float(infer_timing["h2d_ms"]),
                    "execute_ms": float(infer_timing["execute_ms"]),
                    "d2h_ms": float(infer_timing["d2h_ms"]),
                }
            )

            if next_token_id in stop_ids:
                break

            generated_ids.append(next_token_id)
            generated_token_count += 1

            decoded_text = self.tokenizer.decode(
                generated_ids[len(prompt_ids) :], skip_special_tokens=True
            )
            if decoded_text.startswith(emitted_text):
                delta = decoded_text[len(emitted_text) :]
            else:
                delta = decoded_text
            emitted_text = decoded_text

            yield {
                "delta": delta,
                "text": emitted_text,
                "token_id": int(next_token_id),
                "step": step + 1,
                "step_ms": step_ms,
            }

        total_time_s = time.perf_counter() - total_start
        first_token_ms = step_metrics[0]["step_ms"] if step_metrics else 0.0
        avg_token_ms = (
            float(np.mean([item["step_ms"] for item in step_metrics])) if step_metrics else 0.0
        )
        avg_h2d_ms = float(np.mean([item["h2d_ms"] for item in step_metrics])) if step_metrics else 0.0
        avg_execute_ms = (
            float(np.mean([item["execute_ms"] for item in step_metrics])) if step_metrics else 0.0
        )
        avg_d2h_ms = float(np.mean([item["d2h_ms"] for item in step_metrics])) if step_metrics else 0.0
        tokens_per_sec = generated_token_count / total_time_s if total_time_s > 0 else 0.0

        self.last_generation_stats = {
            "prompt_tokens": float(len(prompt_ids)),
            "generated_tokens": float(generated_token_count),
            "tokenize_ms": tokenize_ms,
            "first_token_ms": first_token_ms,
            "avg_token_ms": avg_token_ms,
            "avg_h2d_ms": avg_h2d_ms,
            "avg_execute_ms": avg_execute_ms,
            "avg_d2h_ms": avg_d2h_ms,
            "total_time_s": total_time_s,
            "tokens_per_sec": tokens_per_sec,
            "config": asdict(cfg),
        }

    def generate_text(
        self,
        prompt,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: Optional[bool] = None,
        repetition_penalty: Optional[float] = None,
        config: Optional[GenerationConfig] = None,
    ):
        """
        兼容旧接口的完整文本生成。
        返回: (generated_text, total_inference_time_s)
        """
        response_text = ""
        for chunk in self.stream_generate(
            prompt,
            config=config,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
        ):
            response_text = str(chunk["text"])
        return response_text, float(self.last_generation_stats.get("total_time_s", 0.0))

    def generate_next_token(self, prompt, max_length=None):
        """
        兼容旧测试逻辑：返回下一个 token。
        Returns: (next_token_text, logits, inference_time)
        """
        if max_length is None:
            max_length = self.max_seq_len
        max_length = min(int(max_length), self.max_seq_len)

        inputs = self.tokenizer(
            prompt,
            return_tensors="np",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )
        input_ids = inputs["input_ids"].astype(np.int64, copy=False)
        attention_mask = inputs["attention_mask"].astype(np.int64, copy=False)

        if input_ids.shape[1] < self.max_seq_len:
            pad_width = self.max_seq_len - input_ids.shape[1]
            input_ids = np.pad(input_ids, ((0, 0), (0, pad_width)), mode="constant")
            attention_mask = np.pad(attention_mask, ((0, 0), (0, pad_width)), mode="constant")

        start_time = time.perf_counter()
        outputs, _ = self.inference(input_ids, attention_mask, self.position_ids_buffer)
        logits = self._extract_last_logits(outputs[0], self.max_seq_len, attention_mask)
        next_token_id = int(np.argmax(logits))
        inference_time = time.perf_counter() - start_time
        next_token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=False)
        return next_token_text, logits, inference_time

    def benchmark(
        self,
        prompt_list: Sequence[str],
        warmup_steps: int = 1,
        new_tokens: int = 64,
        config: Optional[GenerationConfig] = None,
    ) -> Dict[str, object]:
        """
        基准测试，返回结构化统计。
        """
        if not prompt_list:
            raise ValueError("prompt_list 不能为空")

        warm_cfg = self._normalize_generation_config(
            GenerationConfig(
                max_new_tokens=8,
                temperature=0.0,
                top_k=0,
                top_p=1.0,
                do_sample=False,
                repetition_penalty=1.0,
            )
        )
        run_cfg = self._normalize_generation_config(config, max_new_tokens=new_tokens)

        for _ in range(max(0, int(warmup_steps))):
            _ = self.generate_text(prompt_list[0], config=warm_cfg)

        runs: List[Dict[str, object]] = []
        for prompt in prompt_list:
            text, _ = self.generate_text(prompt, config=run_cfg)
            stats = dict(self.last_generation_stats)
            stats["prompt"] = prompt
            stats["response_preview"] = text[:120]
            runs.append(stats)

        aggregate = {}
        for key in [
            "first_token_ms",
            "avg_token_ms",
            "tokens_per_sec",
            "tokenize_ms",
            "avg_h2d_ms",
            "avg_execute_ms",
            "avg_d2h_ms",
            "total_time_s",
        ]:
            values = [float(r.get(key, 0.0)) for r in runs]
            aggregate[f"{key}_mean"] = float(np.mean(values)) if values else 0.0
            aggregate[f"{key}_p95"] = float(np.percentile(values, 95)) if values else 0.0

        return {
            "engine": "acl_legacy",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "warmup_steps": int(warmup_steps),
            "run_config": asdict(run_cfg),
            "runs": runs,
            "aggregate": aggregate,
        }

    def save_benchmark_report(self, report: Dict[str, object], output_path: str):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def cleanup(self):
        """释放 ACL 资源。"""
        print("\n[ACL] Cleaning up resources...")

        # 先销毁 data_buffer，再销毁 dataset
        for data_buffer in self.input_data_buffers:
            try:
                acl.destroy_data_buffer(data_buffer)
            except Exception:
                pass
        self.input_data_buffers.clear()

        for data_buffer in self.output_data_buffers:
            try:
                acl.destroy_data_buffer(data_buffer)
            except Exception:
                pass
        self.output_data_buffers.clear()

        if self.input_dataset is not None:
            try:
                acl.mdl.destroy_dataset(self.input_dataset)
            except Exception:
                pass
            self.input_dataset = None

        if self.output_dataset is not None:
            try:
                acl.mdl.destroy_dataset(self.output_dataset)
            except Exception:
                pass
            self.output_dataset = None

        # 释放 device buffer
        for ptr in self.input_buffer_ptrs:
            try:
                acl.rt.free(ptr)
            except Exception:
                pass
        self.input_buffer_ptrs.clear()

        for ptr in self.output_buffer_ptrs:
            try:
                acl.rt.free(ptr)
            except Exception:
                pass
        self.output_buffer_ptrs.clear()

        if self.model_id:
            try:
                acl.mdl.unload(self.model_id)
            except Exception:
                pass
            self.model_id = None

        if self.model_desc:
            try:
                acl.mdl.destroy_desc(self.model_desc)
            except Exception:
                pass
            self.model_desc = None

        if self.stream:
            try:
                acl.rt.destroy_stream(self.stream)
            except Exception:
                pass
            self.stream = None

        if self.context:
            try:
                acl.rt.destroy_context(self.context)
            except Exception:
                pass
            self.context = None

        try:
            acl.rt.reset_device(self.device_id)
        except Exception:
            pass

        try:
            acl.finalize()
        except Exception:
            pass

        print("[ACL] ✓ Cleanup complete")


def main():
    """简单自检。"""
    model_path = "/home/HwHiAiUser/ICT/qwen3_fp16.om"
    tokenizer_path = "/home/HwHiAiUser/ICT/models/qwen3-8b"

    print("=" * 60)
    print("Qwen ACL Inference Smoke Test")
    print("=" * 60)

    engine = None
    try:
        engine = Qwen3ACLInference(model_path, tokenizer_path)
        test_prompts = [
            "你好，请用一句话介绍香橙派 AI Pro。",
            "What is Ascend 310B?",
            "请简要解释什么是边缘 AI。",
        ]

        report = engine.benchmark(test_prompts, warmup_steps=1, new_tokens=32)
        print(json.dumps(report["aggregate"], ensure_ascii=False, indent=2))

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if engine is not None:
            engine.cleanup()


if __name__ == "__main__":
    main()
