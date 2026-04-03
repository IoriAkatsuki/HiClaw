#!/usr/bin/env python3
"""
Streamlit WebUI for Qwen on OrangePi AI Pro.

支持：
1. ACL 旧链路（默认）
2. MindIE HTTP 后端（灰度切换）
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional

import streamlit as st

try:
    import yaml
except Exception:
    yaml = None

# Add path for runtime modules
sys.path.insert(0, "/home/HwHiAiUser/ICT")

try:
    from qwen3_acl_inference import GenerationConfig, Qwen3ACLInference
except Exception as exc:
    Qwen3ACLInference = None
    @dataclass
    class GenerationConfig:  # type: ignore[no-redef]
        max_new_tokens: int = 128
        temperature: float = 0.9
        top_k: int = 40
        top_p: float = 0.9
        do_sample: bool = True
        repetition_penalty: float = 1.0
    _ACL_IMPORT_ERR = str(exc)
else:
    _ACL_IMPORT_ERR = ""

try:
    from mindie_backend import MindIEBackend
except Exception:
    MindIEBackend = None

DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "backend": "acl_legacy",
    "acl": {
        "model_path": "/home/HwHiAiUser/ICT/qwen3_1_7b_fp16.om",
        "tokenizer_path": "/home/HwHiAiUser/ICT/models/qwen3-1.7b",
        "device_id": 0,
        "max_seq_len": 512,
    },
    "mindie": {
        "enabled": False,
        "endpoint": "http://127.0.0.1:1025/v1/chat/completions",
        "model": "qwen3-1.7b",
        "timeout_s": 120,
    },
    "generation_defaults": {
        "max_new_tokens": 128,
        "temperature": 0.9,
        "top_k": 40,
        "top_p": 0.9,
        "do_sample": True,
        "repetition_penalty": 1.0,
    },
}


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def load_runtime_config() -> Dict[str, Any]:
    cfg = deepcopy(DEFAULT_RUNTIME_CONFIG)
    cfg_path = os.getenv("LLM_RUNTIME_CONFIG", "/home/HwHiAiUser/ICT/config/llm_runtime.yaml")
    if not os.path.exists(cfg_path):
        return cfg
    if yaml is None:
        st.warning("未安装 PyYAML，使用内置默认配置。")
        return cfg

    with open(cfg_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if isinstance(loaded, dict):
        deep_update(cfg, loaded)
    return cfg


def init_engine(runtime_cfg: Dict[str, Any]):
    backend = runtime_cfg.get("backend", "acl_legacy")
    err = ""

    if backend == "mindie":
        mindie_cfg = runtime_cfg.get("mindie", {})
        if MindIEBackend is None:
            err = "MindIE 后端模块不可用，自动回退 ACL。"
        elif not mindie_cfg.get("enabled", False):
            err = "mindie.enabled=false，自动回退 ACL。"
        else:
            try:
                engine = MindIEBackend(
                    endpoint=mindie_cfg.get("endpoint", ""),
                    model=mindie_cfg.get("model", "qwen3"),
                    timeout_s=int(mindie_cfg.get("timeout_s", 120)),
                )
                return engine, "mindie", err
            except Exception as exc:
                err = f"MindIE 初始化失败，自动回退 ACL: {exc}"

    # Fallback ACL
    if Qwen3ACLInference is None:
        raise RuntimeError(f"ACL 引擎导入失败: {_ACL_IMPORT_ERR}")

    acl_cfg = runtime_cfg.get("acl", {})
    engine = Qwen3ACLInference(
        acl_cfg.get("model_path"),
        acl_cfg.get("tokenizer_path"),
        device_id=int(acl_cfg.get("device_id", 0)),
        max_seq_len=int(acl_cfg.get("max_seq_len", 512)),
    )
    return engine, "acl_legacy", err


def create_generation_config(sidebar_values: Dict[str, Any]) -> "GenerationConfig":
    return GenerationConfig(
        max_new_tokens=int(sidebar_values["max_new_tokens"]),
        temperature=float(sidebar_values["temperature"]),
        top_k=int(sidebar_values["top_k"]),
        top_p=float(sidebar_values["top_p"]),
        do_sample=bool(sidebar_values["do_sample"]),
        repetition_penalty=float(sidebar_values["repetition_penalty"]),
    )


st.set_page_config(page_title="Qwen Chat (OrangePi)", layout="wide")
st.title("Qwen Chat on OrangePi AI Pro")

runtime_cfg = load_runtime_config()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "engine" not in st.session_state:
    st.session_state.engine = None
if "backend_name" not in st.session_state:
    st.session_state.backend_name = ""
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = {}

if st.session_state.engine is None:
    with st.spinner("加载推理引擎中..."):
        try:
            engine, backend_name, err = init_engine(runtime_cfg)
            st.session_state.engine = engine
            st.session_state.backend_name = backend_name
            if err:
                st.warning(err)
            st.success(f"推理引擎就绪: {backend_name}")
        except Exception as e:
            st.error(f"引擎初始化失败: {e}")

gen_defaults = runtime_cfg.get("generation_defaults", {})
with st.sidebar:
    st.header("Runtime")
    st.write(f"**Backend**: {st.session_state.backend_name or 'N/A'}")

    if st.session_state.backend_name == "acl_legacy":
        model_path = runtime_cfg.get("acl", {}).get("model_path", "")
        st.write(f"**OM**: {os.path.basename(model_path)}")
    elif st.session_state.backend_name == "mindie":
        st.write(f"**Endpoint**: {runtime_cfg.get('mindie', {}).get('endpoint', '')}")

    st.divider()
    st.header("Generation")
    sidebar_values = {
        "max_new_tokens": st.slider(
            "Max new tokens",
            min_value=16,
            max_value=512,
            value=int(gen_defaults.get("max_new_tokens", 128)),
            step=16,
        ),
        "temperature": st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=float(gen_defaults.get("temperature", 0.9)),
            step=0.1,
        ),
        "top_k": st.slider(
            "Top-k",
            min_value=0,
            max_value=200,
            value=int(gen_defaults.get("top_k", 40)),
            step=5,
        ),
        "top_p": st.slider(
            "Top-p",
            min_value=0.0,
            max_value=1.0,
            value=float(gen_defaults.get("top_p", 0.9)),
            step=0.05,
        ),
        "do_sample": st.checkbox(
            "Do sample",
            value=bool(gen_defaults.get("do_sample", True)),
        ),
        "repetition_penalty": st.slider(
            "Repetition penalty",
            min_value=1.0,
            max_value=1.5,
            value=float(gen_defaults.get("repetition_penalty", 1.0)),
            step=0.01,
        ),
    }

    st.divider()
    st.header("Last Metrics")
    metrics = st.session_state.last_metrics or {}
    if metrics:
        st.metric("首 token 延迟 (ms)", f"{metrics.get('first_token_ms', 0.0):.1f}")
        st.metric("平均 token 延迟 (ms)", f"{metrics.get('avg_token_ms', 0.0):.1f}")
        st.metric("吞吐 (tokens/s)", f"{metrics.get('tokens_per_sec', 0.0):.2f}")
        st.caption(
            "H2D/Execute/D2H: "
            f"{metrics.get('avg_h2d_ms', 0.0):.2f}/"
            f"{metrics.get('avg_execute_ms', 0.0):.2f}/"
            f"{metrics.get('avg_d2h_ms', 0.0):.2f} ms"
        )
    else:
        st.caption("暂无生成指标")

    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.last_metrics = {}
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("请输入问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.engine is None:
            st.error("推理引擎不可用。")
        else:
            try:
                response_box = st.empty()
                response_text = ""

                cfg = create_generation_config(sidebar_values)

                # 优先走流式接口
                if hasattr(st.session_state.engine, "stream_generate"):
                    for chunk in st.session_state.engine.stream_generate(prompt, config=cfg):
                        text_now = str(chunk.get("text", ""))
                        if text_now:
                            response_text = text_now
                        else:
                            response_text += str(chunk.get("delta", ""))
                        response_box.markdown(response_text)
                else:
                    response_text, _ = st.session_state.engine.generate_text(prompt, config=cfg)
                    response_box.markdown(response_text)

                if not response_text:
                    response_text = "(空响应)"
                    response_box.markdown(response_text)

                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

                stats = getattr(st.session_state.engine, "last_generation_stats", {}) or {}
                st.session_state.last_metrics = stats
                if stats:
                    st.caption(
                        f"总耗时 {stats.get('total_time_s', 0.0):.2f}s | "
                        f"首 token {stats.get('first_token_ms', 0.0):.1f}ms | "
                        f"吞吐 {stats.get('tokens_per_sec', 0.0):.2f} tok/s"
                    )
            except Exception as e:
                st.error(f"推理异常: {e}")
                import traceback

                st.code(traceback.format_exc())

st.divider()
st.caption(
    "当前支持 ACL 和 MindIE 后端切换；建议先在 ACL 链路做基准，再逐步灰度迁移 MindIE。"
)
