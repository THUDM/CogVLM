from enum import Enum
import streamlit as st
import demo_vqa, demo_vagent
from utils import encode_file_to_base64, templates

st.markdown("<h3>CogAgent & CogVLM Chat Demo</h3>", unsafe_allow_html=True)
st.markdown(
    "<sub>智谱AI 公开在线技术文档: https://lslfd0slxc.feishu.cn/wiki/WvQbwIJ9tiPAxGk8ywDck6yfnof </sub> \n\n <sub> 更多使用方法请参考文档。</sub>",
    unsafe_allow_html=True)


class Mode(str, Enum):
    VQA, VAgent = '💬 VQA', '🧑‍💻 VAgent'


with st.sidebar:
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.0, 1.5, 0.95, step=0.01
    )
    repetition_penalty = st.slider(
        'repetition_penalty', 0.0, 2.0, 1.0, step=0.01
    )
    max_new_token = st.slider(
        'Output length', 10, 1024, 256, step=1
    )
    uploaded_file = st.file_uploader("Choose an image...", type=['.jpg', '.png', '.jpeg'], accept_multiple_files=False)
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
        st.warning("""
                Only support for CogAgent-chat-17B and please choose one template below to get better performance.
            """)
        selected_template = st.selectbox("Select a Template", templates)

match tab:
    case Mode.VQA:
            demo_vqa.main(top_p=top_p,
                          temperature=temperature,
                          prompt_text=prompt_text,
                          metadata=encode_file_to_base64(uploaded_file) if uploaded_file else None,
                          repetition_penalty=repetition_penalty,
                          max_new_tokens=max_new_token
                          )
    case Mode.VAgent:
        demo_vagent.main(top_p=top_p,
                         temperature=temperature,
                         prompt_text=prompt_text,
                         metadata=encode_file_to_base64(uploaded_file) if uploaded_file else None,
                         repetition_penalty=repetition_penalty,
                         max_new_tokens=max_new_token,
                         grounding=grounding,
                         template=selected_template
                         )
    case _:
        st.error(f'Unexpected tab: {tab}')
