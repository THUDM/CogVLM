"""
This is a demo for using Chat Version about CogAgent and CogVLM in WebDEMO
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
This Demo support models in 1 GPU, 1 Batch Only.S trongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
"""

import streamlit as st
from enum import Enum

from utils import encode_file_to_base64, templates
import demo_vqa, demo_vagent

st.markdown("<h3>CogAgent & CogVLM Chat Demo</h3>", unsafe_allow_html=True)
st.markdown(
    "<sub>智谱AI 更多使用方法请参考文档: https://lslfd0slxc.feishu.cn/wiki/WvQbwIJ9tiPAxGk8ywDck6yfnof </sub> \n\n",
    unsafe_allow_html=True)


class Mode(str, Enum):
    VQA, VAgent = '💬 VQA', '🧑‍💻 VAgent'


with st.sidebar:
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.0, 1.0, 0.90, step=0.01
    )
    top_k = st.slider(
        'top_k', 1, 20, 5, step=1
    )
    max_new_token = st.slider(
        'Output length', 1, 2048, 2048, step=1
    )
    uploaded_file = st.file_uploader("Choose an image...", type=['.jpg', '.png', '.jpeg'], accept_multiple_files=False)

    cols = st.columns(2)
    export_btn = cols[0]
    clear_history = cols[1].button("Clear History", use_container_width=True)
    retry = export_btn.button("Retry", use_container_width=True)

prompt_text = st.chat_input(
    'Chat with CogAgent-chat-17B | CogVLM-chat-17B Demo',
    key='chat_input',
)

tab = st.radio(
    'Mode',
    [mode.value for mode in Mode],
    horizontal=True,
    label_visibility='hidden',
)

if tab == Mode.VAgent.value:
    with st.sidebar:
        grounding = st.checkbox("Grounding")
        st.info("""
Only support for CogAgent-chat-17B and please choose one template below to get better performance. and you Just need to write your <TASK>.""")
        selected_template = st.selectbox("Select a Template", templates)

if clear_history or retry:
    prompt_text = ""

match tab:
    case Mode.VQA:
        demo_vqa.main(
            retry=retry,
            top_p=top_p,
            temperature=temperature,
            prompt_text=prompt_text,
            metadata=encode_file_to_base64(uploaded_file) if uploaded_file else None,
            top_k=top_k,
            max_new_tokens=max_new_token
        )
    case Mode.VAgent:
        demo_vagent.main(
            retry=retry,
            top_p=top_p,
            temperature=temperature,
            prompt_text=prompt_text,
            metadata=encode_file_to_base64(uploaded_file) if uploaded_file else None,
            top_k=top_k,
            max_new_tokens=max_new_token,
            grounding=grounding,
            template=selected_template
        )
    case _:
        st.error(f'Unexpected tab: {tab}')
