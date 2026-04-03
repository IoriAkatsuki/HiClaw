#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemma-3 1B WebUI 启动脚本（GPU 推理）

说明：
- 默认使用 HuggingFace 上的官方模型：google/gemma-3-1b-it
- 需要提前在本机执行一次 `huggingface-cli login` 并在网页上同意 Gemma 许可证
- 推理默认走本机 GPU（RTX 4080），也可退回 CPU（速度会明显变慢）
"""

import os
from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr


def load_model():
    """加载 Gemma-3 1B 模型与分词器。"""
    model_id = os.environ.get("GEMMA3_MODEL_ID", "google/gemma-3-1b-it")

    print(f"[INFO] 准备加载模型: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    print(f"[INFO] 模型已加载到设备: {device}")
    return tokenizer, model


tokenizer, model = load_model()


def build_prompt(history: List[Tuple[str, str]], message: str) -> str:
    """
    将历史对话和当前问题整理为模型输入。
    对于 Gemma-3，优先使用 chat_template。
    """
    messages = []
    for user, assistant in history:
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": message})

    # 若 tokenizer 提供 chat_template，则使用；否则退回简单拼接
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # 回退方案：简单拼接
    prompt = ""
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "user":
            prompt += f"用户: {content}\n"
        else:
            prompt += f"助手: {content}\n"
    prompt += "助手: "
    return prompt


def chat_fn(message: str, history: List[Tuple[str, str]], max_new_tokens: int, temperature: float, top_p: float):
    """Gradio ChatInterface 回调函数。"""
    prompt = build_prompt(history, message)
    inputs = tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )
    # 只取新生成部分
    generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response


def main():
    """启动 WebUI。"""
    with gr.Blocks(title="Gemma-3 1B WebUI") as demo:
        gr.Markdown(
            """
            # Gemma-3 1B WebUI

            - 模型：`google/gemma-3-1b-it`
            - 首次使用前请在终端执行：`huggingface-cli login` 并在浏览器中同意 Gemma 许可证
            - 默认使用本机 GPU（如可用），否则回退到 CPU
            """
        )

        with gr.Row():
            max_new_tokens_slider = gr.Slider(
                minimum=16,
                maximum=1024,
                value=256,
                step=16,
                label="max_new_tokens（回复长度）",
            )
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=1.5,
                value=0.7,
                step=0.05,
                label="temperature（发散程度）",
            )
            top_p_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.9,
                step=0.05,
                label="top_p（核采样）",
            )

        chat = gr.ChatInterface(
            fn=chat_fn,
            title="Gemma-3 1B Chat",
            additional_inputs=[max_new_tokens_slider, temperature_slider, top_p_slider],
        )

    # 监听所有网卡，方便局域网访问；端口默认 7860
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
