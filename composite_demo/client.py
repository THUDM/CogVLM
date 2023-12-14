from __future__ import annotations

from threading import Thread

import streamlit as st
import torch
import warnings
import os

from typing import Any, Protocol
from collections.abc import Iterable
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
from transformers import AutoTokenizer, TextIteratorStreamer, AutoModelForCausalLM
from conversation import Conversation

TOOL_PROMPT = 'Answer the following questions as best as you can. You have access to the following tools:'
MODEL_PATH = os.environ.get('MODEL_PATH', '/share/official_pretrains/hf_home/cogagent-chat-hf')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', '/share/official_pretrains/hf_home/vicuna-7b-v1.5')
PT_PATH = os.environ.get('PT_PATH', None)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16
    warnings.warn("Your GPU does not support bfloat16 type, use fp16 instead")


@st.cache_resource
def get_client() -> Client:
    client = HFClient(MODEL_PATH, TOKENIZER_PATH, DEVICE)
    return client


def process_history(history: list[Conversation]):
    history_pairs = []
    query = ""
    last_user_image = None

    user_text = None
    for i, conversation in enumerate(history):
        if conversation.role == conversation.role.USER:
            user_text = conversation.content
            if conversation.image:
                last_user_image = conversation.image

            if i == len(history) - 1:
                query = conversation.content

        else:
            if user_text is not None:
                history_pairs.append((user_text, conversation.content))
                user_text = None
    return query, history_pairs, last_user_image


class Client(Protocol):
    def generate_stream(self,
                        history: list[Conversation],
                        grounding: bool = False,
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        ...


class HFClient(Client):
    def __init__(self, model_path: str, tokenizer_path: str, DEVICE='cpu'):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(DEVICE).eval()

    def generate_stream(self,
                        history: list,
                        grounding: bool = False,
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        """
        chat with CogVLM
        """
        query, history, image = process_history(history)

        print("\n==Input:==\n", query)
        print("\n==History:==\n", history)

        if grounding:
            query += "(with grounding)"

        inputs = self.model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=history,
            images=[image]
        )
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[inputs['images'][0].to(DEVICE).to(torch_type)]],
            'cross_images': [[inputs['cross_images'][0].to(DEVICE).to(torch_type)]] if inputs[
                'cross_images'] else None,
        }
        streamer = TextIteratorStreamer(self.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        parameters['streamer'] = streamer
        gen_kwargs = {**parameters, **inputs}
        with torch.no_grad():
            self.model.generate(**inputs, **parameters)
            thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
            thread.start()
            for next_text in streamer:
                yield TextGenerationStreamResponse(
                    token=Token(
                        id=0,
                        logprob=0,
                        text=next_text,
                        special=False,
                    )
                )
