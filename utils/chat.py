# -*- encoding: utf-8 -*-
'''
@File    :   chat.py
@Time    :   2023/05/08 19:10:08
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

import os
import sys
import re
from functools import partial
from typing import Optional, Tuple, Union, List, Callable, Dict, Any
import requests
from PIL import Image
from io import BytesIO

import torch
from sat.generation.autoregressive_sampling import filling_sequence, BaseStrategy, get_masks_and_position_ids_default
from sat.mpu import get_model_parallel_rank

def process_image(text, text_processor, img_processor, image=None):
    image_position = text.rfind(text_processor.tokenizer.boi) + 5
    if image_position < 5:
        return text, image_position, (None, None)
    # extract path from [IMG][/IMG] using re
    pattern = (text_processor.tokenizer.boi + r"(.*?)" + text_processor.tokenizer.eoi).replace('[', r'\[').replace(']', r'\]')
    image_path = re.findall(pattern, text)
    image_path = image_path[-1] if image_path[-1] else None
    if image is None:
        assert image_path is not None, "image and image_path cannot be both None."
        text = text.replace(image_path, "")
        image_path = image_path.strip()
        # url
        if image_path.startswith("http"):
            response = requests.get(image_path, timeout=10)
            image = Image.open(BytesIO(response.content))
        # local path
        else:
            image = Image.open(image_path)
    if image is not None and isinstance(image, Image.Image):
        pil_img = image.convert('RGB')
        image = img_processor(pil_img) if img_processor is not None else {}
        ret = (image, pil_img)
    else:
        ret = image
    return text, image_position, ret


def chat(image_path, model, text_processor, img_processor,
        query: str, history: List[Tuple[str, str]] = None, image: Image = None,
        max_length: int = 1024, top_p=0.7, top_k=30, temperature=0.95, repetition_penalty=1.2,
        invalid_slices=[], no_prompt=False, force_pil_image=None
        ):
    is_image_mode = image_path or (type(image) is not tuple and image is not None) or (type(image) is tuple and image != (None, None)) or force_pil_image is not None
    if not history:
        history = []
    if is_image_mode and not force_pil_image:
        prompt = "{}{}{}".format(text_processor.tokenizer.boi, image_path if image_path else "", text_processor.tokenizer.eoi)
    else:
        prompt = ""
    if not is_image_mode or not no_prompt:
        prompt += text_processor.history_to_prompt(history, query)
    if force_pil_image is not None:
        image_position = 0
        torch_image = img_processor(force_pil_image) if img_processor is not None else {}
        pil_img = force_pil_image
    else:
        prompt, image_position, (torch_image, pil_img) = process_image(prompt, text_processor, img_processor, image=image)
    if torch_image is not None:
        assert type(torch_image) is dict
        if type(torch_image) is dict:
            for k in torch_image:
                if type(torch_image[k]) is torch.Tensor and torch_image[k].dtype is not torch.int and torch_image[k].dtype is not torch.long:
                    torch_image[k] = torch_image[k].to(next(model.parameters()).dtype)
                if type(torch_image[k]) is torch.Tensor:
                    torch_image[k] = torch_image[k].to(next(model.parameters()).device)
        else:
            torch_image = torch_image.to(next(model.parameters()).dtype).to(next(model.parameters()).device)
        
    if not is_image_mode: # no image
        raise Exception("No image is not supported!")
    else:
        new_prompt = prompt[image_position:]
        if not torch_image or hasattr(text_processor, 'no_eoi'):
            new_prompt = new_prompt.replace(text_processor.tokenizer.eoi, '', 1)
        inputs_dic = text_processor(new_prompt)
        for k in inputs_dic:
            if type(inputs_dic[k]) is torch.Tensor and inputs_dic[k].dtype is not torch.int and inputs_dic[k].dtype is not torch.long:
                inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).dtype)
            if type(inputs_dic[k]) is torch.Tensor:
                inputs_dic[k] = inputs_dic[k].to(next(model.parameters()).device)
        inputs = inputs_dic['input_ids'].to(model.parameters().__next__().device)[0]
    
    if max_length-len(inputs) <=1:
        response = "The prompt exceeds the context length limit, please try again."
        return response, history, (torch_image, pil_img)
    
    seq = torch.cat(
        [inputs, torch.tensor([-1]*(max_length-len(inputs)), device=inputs.device)], dim=0
    )
    strategy = BaseStrategy(temperature=temperature, top_p=top_p, top_k=top_k, end_tokens=[text_processor.tokenizer.eos_token_id],
                            invalid_slices=invalid_slices, repetition_penalty=repetition_penalty)
    get_func = text_processor.get_func(inputs, **inputs_dic) if hasattr(text_processor, 'get_func') else get_masks_and_position_ids_default
    if not is_image_mode:
        inputs = {}
    else:
        inputs = {'vision_'+k:v for k,v in torch_image.items()}
        inputs_dic.pop('input_ids')
        inputs = {**inputs, **inputs_dic}
    output = filling_sequence(
        model, seq,
        batch_size=1,
        get_masks_and_position_ids=get_func,
        strategy=strategy,
        **inputs
    )[0] # drop memory
    
    # ---------------
    # port from inference_glm.py, more general than chat mode
    # clip -1s and fill back generated things into seq
    if type(output) is not list:
        output_list = output.tolist()
    else:
        output_list = output

    response = text_processor.tokenizer.decode(output_list[0])
    # print('original:', response)
    if hasattr(text_processor, 'process_response'):
        response = text_processor.process_response(response)
    response = response.split(text_processor.sep)[-1].strip()
    if get_model_parallel_rank() == 0:
        from utils.parser import parse_response
        parse_response(pil_img, response)
    history = history + [(query, response)]
    return response, history, (torch_image, pil_img)
