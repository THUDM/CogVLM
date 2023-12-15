"""
This is a demo for using CogAgent and CogVLM in CLI
Make sure you have installed vicuna-7b-v1.5 tokenizer model (https://huggingface.co/lmsys/vicuna-7b-v1.5), full checkpoint of vicuna-7b-v1.5 LLM is not required.
In this demo, We us chat template, you can use others to replace such as 'vqa'.
Strongly suggest to use GPU with bfloat16 support, otherwise, it will be slow.
Mention that only one picture can be processed at one conversation, which means you can not replace or insert another picture during the conversation.
"""
import os
import warnings
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

MODEL_PATH = os.environ.get('MODEL_PATH', 'your cogagent-chat-hf path')
TOKENIZER_PATH = os.environ.get('TOKENIZER_PATH', 'your vicuna-7b-v1.5 path')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16
    warnings.warn("Your GPU does not support bfloat16 type, use fp16 instead")

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch_type,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(DEVICE).eval()

while True:
    image_path = input("image path >>>>> ")
    if image_path == "stop":
        break

    image = Image.open(image_path).convert('RGB')
    history = []
    while True:
        query = input("Human:")
        if query == "clear":
            break
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=history,
            images=[image]
        )
        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]],
        }
        if 'cross_images' in input_by_model and input_by_model['cross_images']:
            inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

        # add any transformers params here.
        gen_kwargs = {"max_length": 2048,
                      "temperature": 0.9,
                      "do_sample": False}
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]
            print("\nCog:", response)
        history.append((query, response))

