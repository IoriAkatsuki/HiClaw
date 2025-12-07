from transformers import AutoTokenizer, AutoModelForCausalLM
from ftllm import llm
import streamlit as st
import os

st.set_page_config(page_title="FastLLM Chat", layout="wide")
st.title("Qwen3-8B on OrangePi AIpro (FastLLM)")

@st.cache_resource
def load_model():
    model_path = "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen3-8B-Instruct"
    # Fallback to Qwen2.5 if Qwen3 is not found (for demo stability if download failed)
    if not os.path.exists(model_path):
        model_path = "/home/HwHiAiUser/ICT/qwen25_fastllm/models/qwen/Qwen2.5-1.5B-Instruct"
        st.warning(f"Qwen3-8B not found, falling back to: {model_path}")
    
    st.info(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()
        # Use float16 for 24GB RAM
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
            history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    history.append((msg["content"], ""))
                elif msg["role"] == "assistant" and history:
                    history[-1] = (history[-1][0], msg["content"])
            
            # Stream response
            for chunk in model.stream_response(prompt, history=history):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
