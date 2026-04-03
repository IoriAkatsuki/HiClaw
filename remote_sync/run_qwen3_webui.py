import sys
import os
from pathlib import Path

# 1. Fix for ModuleNotFoundError: No module named 'ftllm'
# Add the directory containing the compiled fastllm python bindings
# We assume the standard structure: ~/ICT/qwen25_fastllm/fastllm_src/build/tools
fastllm_tools_path = os.path.expanduser("~/ICT/qwen25_fastllm/fastllm_src/build/tools")
if os.path.exists(fastllm_tools_path):
    sys.path.insert(0, fastllm_tools_path)
else:
    print(f"Warning: ftllm path {fastllm_tools_path} does not exist.")

try:
    from ftllm import llm
except ImportError:
    # Fallback or error handling if still failing
    import streamlit as st
    st.error("Failed to import 'ftllm'. Please verify 'fastllm_src/build/tools' contains the compiled python bindings.")
    sys.exit(1)

from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st

st.set_page_config(page_title="FastLLM Chat", layout="wide")
st.title("Qwen Chat on OrangePi AIpro (Ascend 310B)")

@st.cache_resource
def load_model():
    # Try to find the model. 
    # Check for Qwen2.5-7B (downloaded as fallback) or Qwen3-8B (if it existed)
    # Based on the `find` command output, the structure seems to vary slightly.
    
    candidates = [
        "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen/Qwen2.5-7B-Instruct", # Standard download path
        "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen2___5-7B-Instruct", # Flattened
        "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen/Qwen2___5-7B-Instruct", # Flattened subdir
        "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen3-8B-Instruct",
        "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen2.5-1.5B-Instruct"
    ]
    
    model_path = None
    for p in candidates:
        if os.path.exists(p):
            model_path = p
            break
    
    if not model_path:
        st.error("No model found. Please run the download script first.")
        return None

    st.info(f"Loading model from: {model_path}")
    
    try:
        # Load using Transformers first (easier compatibility)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()
        
        # Convert to FastLLM
        # dtype="float16" is crucial for 24GB RAM and best performance/quality
        model = llm.from_hf(hf_model, tokenizer=tokenizer, dtype="float16")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

if "messages" not in st.session_state:
    st.session_state.messages = []

model = load_model()

if model:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Prepare history for fastllm
            # FastLLM usually expects [(user, bot), (user, bot)] tuples for history
            history = []
            for i in range(0, len(st.session_state.messages)-1, 2):
                if i+1 < len(st.session_state.messages):
                    u = st.session_state.messages[i]
                    b = st.session_state.messages[i+1]
                    if u["role"] == "user" and b["role"] == "assistant":
                        history.append((u["content"], b["content"]))
            
            try:
                # Stream response
                for chunk in model.stream_response(prompt, history=history):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"Generation failed: {e}")
