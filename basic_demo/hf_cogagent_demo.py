import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
tokenizer = LlamaTokenizer.from_pretrained('/share/official_pretrains/hf_home/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    '/share/home/lqs/sat_to_huggingface/cogagent-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to('cuda').eval()


# chat example
query = 'Describe this image in details.'
image = Image.open('/share/home/lqs/vlb_main/fewshot-data/kobe.png').convert('RGB')
inputs = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[image])  # chat mode
inputs = {
    'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
    'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
    'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
    'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
    'cross_images': [[inputs['cross_images'][0].to('cuda').to(torch.bfloat16)]],
}
gen_kwargs = {"max_length": 512, "do_sample": False}
with torch.no_grad():
    outputs = model.generate(**inputs, **gen_kwargs)
    outputs = outputs[:, inputs['input_ids'].shape[1]:]
    print(tokenizer.decode(outputs[0]))