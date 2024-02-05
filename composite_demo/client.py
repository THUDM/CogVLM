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

# Check if GPU supports bfloat16

if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16
    warnings.warn("Your GPU does not support bfloat16 type, use fp16 instead")

# if you use all of Our model, include cogagent-chat cogvlm-chat cogvlm-grounding and put it in different devices, you can do like this.
models_info = {
    'tokenizer': {
        'path': os.environ.get('TOKENIZER_PATH', 'lmsys/vicuna-7b-v1.5'),
    },
    'agent_chat': {
        'path': os.environ.get('MODEL_PATH_AGENT_CHAT', 'THUDM/cogagent-chat-hf'),
        'device': ['cuda:0']
    },
    'vlm_chat': {
        'path': os.environ.get('MODEL_PATH_VLM_CHAT', 'THUDM/cogvlm-chat-hf'),
        'device': ['cuda:3']
    },
    'vlm_grounding': {
        'path': os.environ.get('MODEL_PATH_VLM_GROUNDING','THUDM/cogvlm-grounding-generalist-hf'),
        'device': ['cuda:6']
    }
}


# if you just use one model, use like this
# models_info = {
#     'tokenizer': {
#         'path': os.environ.get('TOKENIZER_PATH', 'lmsys/vicuna-7b-v1.5'),
#     },
#     'agent_chat': {
#         'path': os.environ.get('MODEL_PATH_AGENT_CHAT', 'THUDM/cogagent-chat-hf'),
#         'device': ['cuda:0']
#     },



@st.cache_resource
def get_client() -> Client:
    client = HFClient(models_info)
    return client


def process_history(history: list[Conversation]):
    """
        Process the input history to extract the query and the history pairs.
        Args:
            History(list[Conversation]): A list of Conversation objects representing all conversations.
        Returns:
            query(str): The current user input string.
            history_pairs(list[(str,str)]): A list of (user, assistant) pairs.
            last_user_image(Image): The last user image. Only the latest image.

    """
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
                        model_use: str = 'agent_chat',
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        ...


class HFClient(Client):
    """
        The HFClient class manages the interaction with various large language models
        for text generation tasks. It supports handling multiple models, each designated
        for a specific task like chatting or grounding.

        Args:
            models_info (dict): A dictionary containing the configuration for each model.
                The dictionary format is:
                    - 'tokenizer': Path and settings for the tokenizer.
                    - 'agent_chat': Path and settings for the CogAgent-chat-18B model.
                    - 'vlm_chat': Path and settings for the CogVLM-chat-17B model.
                    - 'vlm_grounding': Path and settings for the CogVLM-grounding-17B model.

        The class loads each model based on the provided information and assigns it to the
        specified CUDA device. It also handles the tokenizer used across all models.
        """
    def __init__(self, models_info):
        self.models = {}
        self.tokenizer = AutoTokenizer.from_pretrained(models_info['tokenizer']['path'], trust_remote_code=True)
        for model_name, model_info in models_info.items():
            if model_name != 'tokenizer':
                self.models[model_name] = []
                for device in model_info['device']:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_info['path'],
                        torch_dtype=torch_type,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,
                    ).to(device).eval()
                    self.models[model_name].append(model)

    def select_best_gpu(self, model_name):
        min_memory_used = None
        selected_model = None

        for model in self.models[model_name]:
            device = next(model.parameters()).device
            mem_used = torch.cuda.memory_allocated(device=device)

            if min_memory_used is None or mem_used < min_memory_used:
                min_memory_used = mem_used
                selected_model = model

        return selected_model

    def generate_stream(self,
                        history: list,
                        grounding: bool = False,
                        model_use: str = 'agent_chat',
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        """
        Generates a stream of text responses based on the input history and selected model.

        This method facilitates a chat-like interaction with the models. Depending on the
        model selected and whether grounding is enabled, it alters the behavior of the text
        generation process.

        Args:
            history (list[Conversation]): A list of Conversation objects representing the
                dialogue history.
            grounding (bool, optional): A flag to indicate whether grounding should be used
                in the generation process. Defaults to False.
            model_use (str, optional): The key name of the model to be used for the generation.
                Defaults to 'agent_chat'.
            **parameters (Any): Additional parameters that may be required for the generation
                process.

        Yields:
            Iterable[TextGenerationStreamResponse]: A stream of text generation responses, each
            encapsulating a generated piece of text.

        The method selects the appropriate model based on `model_use`, processes the input
        history, and feeds it into the model to generate text. It uses threading to handle
        the generation process efficiently.
        """
        query, history, image = process_history(history)
        if grounding:
            query += "(with grounding)"

        model = self.select_best_gpu(model_use)
        device = next(model.parameters()).device

        # Print user input info

        print("\n== Input ==\n", query)
        print("\n==History==\n", history)
        print("\n== Model ==\n\n", model.config.name_or_path)
        print("\n== Device ==\n\n", device)

        input_by_model = model.build_conversation_input_ids(
            self.tokenizer,
            query=query,
            history=history,
            images=[image]
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(device),
            'images': [[input_by_model['images'][0].to(device).to(torch_type)]],
        }

        # CogVLM model do not have param 'cross_images', Only CogAgent have.

        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(device).to(torch_type)]]

        # Use TextIteratorStreamer for streaming generation like huggingface.

        streamer = TextIteratorStreamer(self.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True)
        parameters['streamer'] = streamer
        gen_kwargs = {**parameters, **inputs}
        with torch.no_grad():
            thread = Thread(target=model.generate, kwargs=gen_kwargs)
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
