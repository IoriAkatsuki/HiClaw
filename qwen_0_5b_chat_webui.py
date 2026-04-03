#!/usr/bin/env python3
# 基于 mindnlp 的 Llama Chat WebUI，在香橙派 AI Pro 上运行，供上位机通过 ict.local 访问。

import gradio as gr
import mindspore
from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer
from mindnlp.transformers import TextIteratorStreamer
from threading import Thread


# -------------------- 运行时补丁：修复 mindtorch 未实现 isin 算子问题 --------------------
def _patch_isin_for_mindtorch():
    """
    修补 transformers.pytorch_utils.isin_mps_friendly，避免调用 mindtorch 未实现的 torch.isin。

    思路：
    - 原实现内部直接调用 torch.isin，当后端被 mindtorch 接管且在 CPU 上时会抛出
      `RuntimeError: No implementation for function: isin on CPU.`
    - 我们在这里用 numpy 实现一个等价版本，仅用于 _prepare_special_tokens 中的小张量比较，
      性能影响可以忽略。
    """
    try:
        import transformers
        import torch
        import numpy as np
    except Exception as exc:  # 环境异常直接跳过，不影响后续加载
        print(f"[WARN] isin 补丁导入依赖失败: {exc}")
        return

    try:
        from transformers import pytorch_utils
    except Exception as exc:
        print(f"[WARN] 无法导入 transformers.pytorch_utils: {exc}")
        return

    def _safe_isin_mps_friendly(elements, test_elements):
        """
        使用 numpy 计算 isin，返回与 torch.isin 相同形状的 bool Tensor。

        仅用于少量 special token，比原生实现更通用，避免后端算子不支持。
        """
        try:
            e_np = elements.detach().cpu().numpy()
            t_np = test_elements.detach().cpu().numpy()
            mask_np = np.isin(e_np, t_np)
            mask = torch.from_numpy(mask_np).to(elements.device)
            return mask
        except Exception as inner_exc:
            # 兜底实现：逐元素比较，保证功能正确，不依赖 torch.isin
            print(f"[WARN] numpy isin 失败，使用兜底实现: {inner_exc}")
            try:
                out = torch.zeros_like(elements, dtype=torch.bool)
                for v in test_elements.view(-1):
                    out |= elements == v
                return out
            except Exception as final_exc:
                print(f"[ERROR] isin 兜底实现失败，返回全 False: {final_exc}")
                return torch.zeros_like(elements, dtype=torch.bool)

    # 替换 transformers 内部使用的实现
    pytorch_utils.isin_mps_friendly = _safe_isin_mps_friendly
    transformers.pytorch_utils.isin_mps_friendly = _safe_isin_mps_friendly
    print("[INFO] 已安装 isin_mps_friendly 补丁，避免 mindtorch CPU 上未实现的 isin 算子错误。")


_patch_isin_for_mindtorch()

# 这里选择一个体量接近的 Llama 系列聊天模型，避免显存压力过大。
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

system_prompt = "You are a helpful and friendly Llama-based chatbot."

print("[INFO] Llama WebUI 脚本启动，准备加载模型...")
print(f"[INFO] 加载 {MODEL_ID} 模型，请耐心等待首次下载...")

# 加载 tokenizer 和模型（从 HuggingFace）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    ms_dtype=mindspore.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    ms_dtype=mindspore.float16,
)


def build_input_from_chat_history(chat_history, msg: str):
    """将 Gradio 的 history 转为 chat_template 需要的 messages，过滤 None。"""
    messages = [{"role": "system", "content": system_prompt}]
    for user_msg, ai_msg in chat_history or []:
        if user_msg:
            messages.append({"role": "user", "content": str(user_msg)})
        if ai_msg:
            messages.append({"role": "assistant", "content": str(ai_msg)})
    messages.append({"role": "user", "content": str(msg)})
    return messages


# 推理函数

def predict(message, history):
    print(f"[INFO] 收到对话请求，history 条数: {len(history or [])}")
    messages = build_input_from_chat_history(history, message)
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="ms",
        tokenize=True,
    )
    streamer = TextIteratorStreamer(
        tokenizer,
        timeout=300,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generate_kwargs = dict(
        input_ids=input_ids,
        streamer=streamer,
        max_new_tokens=512,
        # MindTorch 在 CPU 上未实现 torch.multinomial，关闭采样走贪心解码路径以避免报错。
        do_sample=False,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if "</s>" in partial_message:
            break
        yield partial_message


demo = gr.ChatInterface(
    fn=predict,
    title="Llama Chat (MindNLP on OrangePi)",
    description="在香橙派 AI Pro 上运行的 TinyLlama-1.1B-Chat，首次回答会比较慢，请耐心等待。",
    examples=[
        "你是谁？",
        "你能做什么？",
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)
