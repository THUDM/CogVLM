"""
This is a demo using the chat version about CogAgent and CogVLM in WebDEMO

Make sure you have installed the vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5),
and a full checkpoint of vicuna-7b-v1.5 LLM is not required.

Mention that only one image can be processed in a conversation, which means you cannot replace or insert another image
during the conversation.


The models_info parameter is explained as follows
   tokenizer: tokenizer model using vicuna-7b-v1.5 model
   agent_chat: Use the CogAgent-chat-18B model to complete the conversation task
   vlm_chat: Use the CogVLM-chat-17B model to complete the conversation task
   vlm_grounding: Use CogVLM-grounding-17B model to complete the Grounding task

Web Demo user operation logic is as follows:
    CogVLM-Chat -> grounding? - yes -> Choose a template -> CogVLM-grounding-17B
                              - no  -> CogVLM-chat-17B (without grounding)

    CogAgent-Chat  -> CogAgent-chat-18B (Only QA,without Grounding)

    CogAgent-Agent -> CogAgent-chat-18B
                   -> Choose a template -> grounding? - yes -> prompt + (with grounding)
                                                      - no  -> prompt

    CogAgent-vqa-hf are not included in this demo, but you can use it in the same way as CogAgent-chat-18B
    and used it in CogAgent-Chat
"""

import streamlit as st

st.set_page_config(
    page_title="CogVLM & CogAgent Demo",
    page_icon=":robot:",
    layout='centered',
    initial_sidebar_state='expanded',
)

from enum import Enum
from utils import encode_file_to_base64, templates_agent_cogagent, template_grounding_cogvlm
import demo_chat_cogvlm, demo_agent_cogagent, demo_chat_cogagent

st.markdown("<h3>CogAgent & CogVLM Chat Demo</h3>", unsafe_allow_html=True)
st.markdown(
    "<sub>更多使用方法请参考文档: https://lslfd0slxc.feishu.cn/wiki/WvQbwIJ9tiPAxGk8ywDck6yfnof \n\n 请根据文档的引导说明来尝试demo，以便理解demo的布局设计 </sub> \n",
    unsafe_allow_html=True)


class Mode(str, Enum):
    CogVLM_Chat, CogAgent_Chat, CogAgent_Agent = '💬CogVLM-Chat', '🧑‍💻 CogAgent-Chat', '💡 CogAgent-Agent'


with st.sidebar:
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.01, 1.0, 0.90, step=0.01
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
    'Chat with CogAgent | CogVLM',
    key='chat_input',
)

tab = st.radio(
    'Mode',
    [mode.value for mode in Mode],
    horizontal=True,
    label_visibility='hidden',
)

selected_template_grounding_cogvlm = ""
with st.sidebar:
    grounding = st.checkbox("Grounding")
    if tab == Mode.CogVLM_Chat or tab == Mode.CogAgent_Chat:
        if grounding:
            selected_template_grounding_cogvlm = st.selectbox("Template For Grounding", template_grounding_cogvlm)

if tab == Mode.CogAgent_Agent:
    with st.sidebar:
        selected_template_agent_cogagent = st.selectbox("Template For Agent", templates_agent_cogagent)

if clear_history or retry:
    prompt_text = ""

match tab:
    case Mode.CogVLM_Chat:
        st.info("This option uses cogvlm-chat and cogvlm-grounding model.")
        if uploaded_file is not None:
            demo_chat_cogvlm.main(
                retry=retry,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                prompt_text=prompt_text,
                metadata=encode_file_to_base64(uploaded_file),
                max_new_tokens=max_new_token,
                grounding=grounding,
                template=selected_template_grounding_cogvlm
            )
        else:
            st.error(f'Please upload an image to start')

    case Mode.CogAgent_Chat:
        st.info("This option uses cogagent-chat model.")
        if uploaded_file is not None:
            demo_chat_cogagent.main(
                retry=retry,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                prompt_text=prompt_text,
                metadata=encode_file_to_base64(uploaded_file),
                max_new_tokens=max_new_token,
                grounding=grounding,
                template=selected_template_grounding_cogvlm
            )
        else:
            st.error(f'Please upload an image to start')

    case Mode.CogAgent_Agent:
        st.info("This option uses cogagent-chat model with agent template.")
        if uploaded_file is not None:
            demo_agent_cogagent.main(
                retry=retry,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                prompt_text=prompt_text,
                metadata=encode_file_to_base64(uploaded_file),
                max_new_tokens=max_new_token,
                grounding=grounding,
                template=selected_template_agent_cogagent
            )
        else:
            st.error(f'Please upload an image to start')
    case _:
        st.error(f'Unexpected tab: {tab}')
