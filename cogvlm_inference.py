import argparse
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer
import os
import json
import shortuuid
import requests
from io import BytesIO

def parse_args():
    parser = argparse.ArgumentParser(description="Process images for captioning using pre-trained models.")
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing images for captioning")
    parser.add_argument("--url_file", type=str, help="Path to the file containing image URLs")
    parser.add_argument("--prompts_file", type=str, help="Optional path to a JSONL file containing prompts for each image.")
    parser.add_argument("--from_pretrained", type=str, default="THUDM/cogvlm-grounding-generalist-hf", help="Pretrained model identifier or path")
    parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help="Tokenizer identifier or path")
    parser.add_argument("--quant", type=int, choices=[4, 8, 16], help="Quantization bits")
    parser.add_argument("--query", type=str, default="Describe the image accurately and in detail.", help="Default query for captioning")
    parser.add_argument("--fp16", action="store_true", help="Enable half-precision floating point (16-bit)")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 precision floating point (16-bit)")
    return parser.parse_args()

def load_model(from_pretrained, use_bfloat16, quantization=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if use_bfloat16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        from_pretrained,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        load_in_4bit=quantization is not None,
        trust_remote_code=True,
    ).eval()
    return model, device

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

def load_images(args):
    if args.folder_path:
        image_files = {filename: os.path.join(args.folder_path, filename)
                       for filename in os.listdir(args.folder_path)
                       if filename.endswith((".jpg", ".png"))}
        images = {name: Image.open(path).convert("RGB") for name, path in image_files.items()}
        return images
    elif args.url_file:
        with open(args.url_file, 'r') as file:
            data = json.load(file)
            images = {item['file_name']: Image.open(BytesIO(requests.get(item['coco_url']).content)).convert("RGB")
                      for item in data['images']}
        return images
    else:
        raise ValueError("No valid input source provided. Please specify a folder path or a JSON file.")

def main():
    args = parse_args()
    model, DEVICE = load_model(args.from_pretrained, args.bf16, args.quant)
    images = load_images(args)
    tokenizer = LlamaTokenizer.from_pretrained(args.local_tokenizer)
    if args.bf16:
        torch_type = torch.bfloat16
    else:
        torch_type = torch.float16
    if args.prompts_file:
        prompts_dict = load_prompts(args.prompts_file)

    with open("cogvlm_outputs.jsonl", "w") as ans_file:
        for filename, image in images.items():
            queries = prompts_dict.get(filename, [args.query]) if args.prompts_file else [args.query]
            
            for query in queries:
                input_by_model = model.build_conversation_input_ids(
                tokenizer, query=query, history=[], images=[image]
                )
                inputs = {
                    "input_ids": input_by_model["input_ids"].unsqueeze(0).to(DEVICE),
                    "token_type_ids": input_by_model["token_type_ids"].unsqueeze(0).to(DEVICE),
                    "attention_mask": input_by_model["attention_mask"].unsqueeze(0).to(DEVICE),
                    "images": [[input_by_model["images"][0].to(DEVICE).to(torch_type)]]
                    if image is not None
                    else None,
                }
                with torch.no_grad():
                    outputs = model.generate(**inputs)
                    outputs = outputs[:, inputs["input_ids"].shape[1] :]
                    response = tokenizer.decode(outputs[0])
                    response = response.split("</s>")[0]
                    print(query)
                    print("\nCog:", response)

                ans_id = shortuuid.uuid()
                result = {
                    "file": filename,
                    "prompt": query,
                    "text": response,
                    "answer_id": ans_id
                }
                ans_file.write(json.dumps(result) + '\n')

    print("Results written to JSON file successfully.")

if __name__ == "__main__":
    main()