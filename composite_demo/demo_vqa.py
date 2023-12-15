import streamlit as st
import base64

from PIL import Image
from io import BytesIO
from streamlit.delta_generator import DeltaGenerator
from client import get_client
from utils import images_are_same
from conversation import Conversation, Role, postprocess_image

client = get_client()


def append_conversation(
        conversation: Conversation,
        history: list[Conversation],
        placeholder: DeltaGenerator | None = None,
) -> None:
    history.append(conversation)
    conversation.show(placeholder)


def main(top_p: float,
         temperature: float,
         prompt_text: str,
         metadata: str,
         repetition_penalty: float,
         max_new_tokens: int):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    history: list[Conversation] = st.session_state.chat_history

    for conversation in history:
        conversation.show()
    if prompt_text:

        image = Image.open(BytesIO(base64.b64decode(metadata))).convert('RGB') if metadata else None
        image.thumbnail((1120, 1120))
        image_input = image
        if history and image:
            last_user_image = next(
                (conv.image for conv in reversed(history) if conv.role == Role.USER and conv.image), None)
            if last_user_image and images_are_same(image, last_user_image):
                image_input = None
            else:
                st.session_state.chat_history = []
                history = []

        # Set conversation
        user_conversation = Conversation(role=Role.USER, content_show=prompt_text.strip(), image=image_input)
        append_conversation(user_conversation, history)
        placeholder = st.empty()
        assistant_conversation = placeholder.chat_message(name="assistant", avatar="assistant")
        assistant_conversation = assistant_conversation.empty()

        # steam Answer
        output_text = ''
        for response in client.generate_stream(
                history=history,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
        ):
            output_text += response.token.text
            assistant_conversation.markdown(output_text.strip() + '▌')

        ## Final Answer with image.
        print("\n==Output:==\n", output_text)
        content_output, image_output = postprocess_image(output_text, image)
        assistant_conversation = Conversation(role=Role.ASSISTANT, content=content_output, image=image_output)
        append_conversation(
            conversation=assistant_conversation,
            history=history,
            placeholder=placeholder.chat_message(name="assistant", avatar="assistant")
        )
