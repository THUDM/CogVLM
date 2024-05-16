# This script opens each image file in a specified folder and generates a caption for it as a .txt file. Built from cogvlm demo repo as the base.

import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import os
import json
import shortuuid

# Create an argument parser to handle command line arguments
parser = argparse.ArgumentParser()

# Define the command line arguments

# Required argument: Path to the folder containing images
parser.add_argument(
    "--folder_path",
    type=str,
    required=True,
    help="Path to the folder containing images for captioning",
)

# Optional arguments
parser.add_argument(
    "--quant", type=int, choices=[4, 8, 16], help="Number of quantization bits"
)
parser.add_argument(
    "--query",
    type=str,
    default="Describe the image accurately and in detail, capturing descriptions of the image and any text.",
    required=False,
    help="Query to pass to the model for captioning, use default query otherwise (ex: Describe what you see in the image)",
)
parser.add_argument(
    "--system",
    type=str,
    default="An image captioning chat between a USER and an ASSISTANT. USER: {} ASSISTANT:",
    required=False,
    help="Default prompt to pass to the model for captioning (ex: An image captioning chat between a USER and an ASSISTANT. USER: <--query> ASSISTANT:)",
)


# ex: THUDM/cogvlm-grounding-generalist-hf, THUDM/cogagent-vqa-hf, THUDM/cogagent-chat-hf, etc
parser.add_argument(
    "--from_pretrained",
    type=str,
    default="THUDM/cogvlm-grounding-generalist-hf",
    help="Path to the pretrained checkpoint",
)
parser.add_argument(
    "--local_tokenizer",
    type=str,
    default="lmsys/vicuna-7b-v1.5",
    help="Path to the tokenizer",
)
parser.add_argument(
    "--fp16", action="store_true", help="Enable half-precision floating point (16-bit)"
)
parser.add_argument(
    "--bf16",
    action="store_true",
    help="Enable bfloat16 precision floating point (16-bit)",
)
parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens")

# Below arguments require do_sample to be True
parser.add_argument(
    "--do_sample",
    action="store_true",
    default=False,
    help="Whether to use sampling during generation",
)
parser.add_argument(
    "--top_p", type=float, default=0.4, help="Top p for nucleus sampling"
)
parser.add_argument("--top_k", type=int, default=1, help="Top k for top k sampling")
parser.add_argument(
    "--temperature", type=float, default=0.8, help="Temperature for sampling"
)

parser.add_argument(
    "--prompts_file", type=str, default=None, help="Optional path to a JSONL file containing prompts for each image.")

args = parser.parse_args()

def load_prompts(prompts_file):
    prompt_dict = {}
    with open(prompts_file, 'r') as file:
        for line in file:
            data = json.loads(line)
            image_name = data['image']
            prompt_text = data['text']

            if image_name not in prompt_dict:
                prompt_dict[image_name] = []

            if prompt_text not in prompt_dict[image_name]:
                prompt_dict[image_name].append(prompt_text)

    return prompt_dict

if args.prompts_file:
    prompts_dict = load_prompts(args.prompts_file)
    
MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print(
    "========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE)
)

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        args.from_pretrained,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        trust_remote_code=True,
    ).eval()
else:
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.from_pretrained,
            torch_dtype=torch_type,
            low_cpu_mem_usage=True,
            load_in_4bit=args.quant is not None,
            trust_remote_code=True,
        )
        .to(DEVICE)
        .eval()
    )

history = []
text_only_template = args.system

# Preallocate the history list
history = []


image_files = [
    filename
    for filename in os.listdir(args.folder_path)
    if filename.endswith((".jpg", ".png"))
]
# Process each image file
with open(os.path.join(args.folder_path,"pope_outputs.jsonl"), "w") as ans_file:
    for filename in image_files:
        image = Image.open(os.path.join(args.folder_path, filename)).convert("RGB")
        
        # Determine the prompt to use
        if args.prompts_file and filename in prompts_dict:
            queries = prompts_dict[filename]
        else:
            queries = [args.query]
        
        for query in queries:
            input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=history, images=[image]
            )
            inputs = {
                "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
                "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
                "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
                "images": [[input_by_model["images"][0].to(DEVICE).to(torch_type)]]
                if image is not None
                else None,
            }
            if "cross_images" in input_by_model and input_by_model["cross_images"]:
                inputs["cross_images"] = [
                    [input_by_model["cross_images"][0].to(DEVICE).to(torch_type)]
                ]

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
            }

            if args.do_sample:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p
                gen_kwargs["top_k"] = args.top_k

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs["input_ids"].shape[1] :]
                response = tokenizer.decode(outputs[0])
                response = response.split("</s>")[0]
                print("\nCog:", response)
            history.append((query, response))

            ans_id = shortuuid.uuid()
            result = {
                "file": filename,
                "prompt": query,
                "text": response,
                "answer_id": ans_id
            }

            ans_file.write(json.dumps(result) + '\n')    
            history.clear()
                
print("Results written to JSON file successfully.")