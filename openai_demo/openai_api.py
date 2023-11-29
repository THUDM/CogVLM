import os
import gc
import time
from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union, Tuple, Optional
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import AutoModelForCausalLM, LlamaTokenizer, PreTrainedModel, PreTrainedTokenizer, \
    TextIteratorStreamer
import requests
from PIL import Image
from io import BytesIO
import base64

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/cogvlm-chat')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", 'lmsys/vicuna-7b-v1.5')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@asynccontextmanager
async def lifespan(app: FastAPI):  # collects GPU memory
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ImageUrl(BaseModel):
    url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentItem = Union[TextContent, ImageUrlContent]


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None


class ChatMessageResponse(BaseModel):  # 模型回复的字段
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Additional parameters
    repetition_penalty: Optional[float] = 1.0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="cogvlm-chat-17b")
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
    )

    if request.stream:
        generate = predict(request.model, gen_params)
        return EventSourceResponse(generate, media_type="text/event-stream")
    response = generate_cogvlm(model, tokenizer, gen_params)

    usage = UsageInfo()

    message = ChatMessageResponse(
        role="assistant",
        content=response["text"],
    )
    logger.debug(f"==== message ====\n{message}")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
    )
    task_usage = UsageInfo.model_validate(response["usage"])
    for usage_key, usage_value in task_usage.model_dump().items():
        setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion", usage=usage)


async def predict(model_id: str, params: dict):
    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_cogvlm(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode
        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))


def generate_cogvlm(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    for response in generate_stream_cogvlm(model, tokenizer, params):
        pass
    return response


def process_history_and_images(messages: List[ChatMessageInput]) -> Tuple[
    Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:
    """
    Process history messages to extract text, identify the last user query,
    and convert base64 encoded image URLs to PIL images.

    :param messages: List of ChatMessageInput objects.
    :return: A tuple of three elements:
             - The last user query as a string.
             - Text history formatted as a list of tuples for the model.
             - List of PIL Image objects extracted from the messages.
    """
    formatted_history = []
    image_list = []
    last_user_query = ''

    for i, message in enumerate(messages):
        role = message.role
        content = message.content

        if isinstance(content, list):  # text
            text_content = ' '.join(item.text for item in content if isinstance(item, TextContent))
        else:
            text_content = content
        
        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if image_url.startswith("data:image/jpeg;base64,"):
                        base64_encoded_image = image_url.split("data:image/jpeg;base64,")[1]
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(BytesIO(image_data)).convert('RGB')
                        image_list.append(image)
        
        if role == 'user':
            if i == len(messages) - 1:  # 最后一条用户消息
                last_user_query = text_content
            else:
                formatted_history.append((text_content, ''))
        elif role == 'assistant':
            if formatted_history:
                if formatted_history[-1][1] != '':
                    assert False, f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, f"assistant reply before user"
        else:
            assert False, f"unrecognized role: {role}"

    return last_user_query, formatted_history, image_list


@torch.inference_mode()
def generate_stream_cogvlm(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, params: dict):
    messages = params["messages"]
    temperature = float(params.get("temperature", 1.0))
    repetition_penalty = float(params.get("repetition_penalty", 1.0))
    top_p = float(params.get("top_p", 1.0))
    max_new_tokens = int(params.get("max_tokens", 256))
    echo = params.get("echo", True)
    query, history, image_list = process_history_and_images(messages)

    logger.debug(f"==== request ====\n{query}")

    #  only can slove the latest picture
    inputs = model.build_conversation_input_ids(tokenizer, query=query, history=history, images=[image_list[-1]])


    inputs = {
        'input_ids': inputs['input_ids'].unsqueeze(0).to(DEVICE),
        'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(DEVICE),
        'attention_mask': inputs['attention_mask'].unsqueeze(0).to(DEVICE),
        'images': [[inputs['images'][0].to(DEVICE).to(torch.bfloat16)]]
    }
    input_echo_len = len(inputs["input_ids"][0])
    streamer = TextIteratorStreamer(tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        "repetition_penalty": repetition_penalty,
        "max_new_tokens": max_new_tokens,
        "do_sample": True if temperature > 1e-5 else False,
        "top_p": top_p,
        'streamer': streamer,
    }
    if temperature > 1e-5:
        gen_kwargs["temperature"] = temperature

    total_len = 0
    generated_text = ""
    with torch.no_grad():
        model.generate(**inputs, **gen_kwargs)
        for next_text in streamer:
            generated_text += next_text
            yield {
                "text": generated_text,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": total_len - input_echo_len,
                    "total_tokens": total_len,
                },
            }
    ret = {
        "text": generated_text,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": total_len - input_echo_len,
            "total_tokens": total_len,
        },
    }
    yield ret


gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    tokenizer = LlamaTokenizer.from_pretrained(
        TOKENIZER_PATH,
        trust_remote_code=True)
    # AMD, NVIDIA GPU can use BF16 Precision
    if 'cuda' in DEVICE:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        ).to(DEVICE).eval()
    # CPU, Intel GPU and other GPU can use Float16 Precision Only
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True).float().to(DEVICE).eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)
