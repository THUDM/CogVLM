import argparse

from transformers import AutoConfig, AutoModelForCausalLM
from peft import PeftModel
import torch

from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ConversationDataset


def get_args():
    parser = argparse.ArgumentParser()
    # model parameters
    parser.add_argument("--model_dir",
                        type=str,
                        default="/share/home/zyx/Models/cogagent-chat-hf",
                        help="Directory where the model is stored")
    parser.add_argument("--tokenizer_dir",
                        type=str,
                        default="/share/official_pretrains/hf_home/vicuna-7b-v1.5",
                        help="Directory for the tokenizer")
    parser.add_argument("--lora_dir",
                        type=str,
                        default="script/output/final_model",
                        help="Directory for the tokenizer")

    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/share/home/zyx/Dataset/new_dataset",
                        help="Directory for the dataset")

    parser.add_argument("--device",
                        type=str,
                        default="cuda:0",
                        help="Device to run the model on")

    parser.add_argument("--max_length",
                        type=int,
                        default=512,
                        help="Frequency of saving the model after epochs")

    return parser.parse_args()


def test(model, tokenizer, test_dataloader):
    for step, batch in enumerate(tqdm(test_dataloader, desc="Testing", position=0, leave=True)):
        with torch.no_grad():
            input_ids = batch['input_ids']
            question = tokenizer.decode(input_ids[input_ids != 0])

            outputs = model.generate(**batch, max_new_tokens=args.max_length)

            outputs = outputs[:, input_ids.shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("</s>")[0]

            labels = batch['labels']
            response_labels = tokenizer.decode(labels[labels != -100])

            print("**************")
            print("Test Question:", question, end="\n\n")
            print("Model Output:", response, end="\n\n")
            print("Ground Truth Output:", response_labels)


def infer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(args.device)

    model = PeftModel.from_pretrained(model, args.lora_dir).to(args.device)

    test_dataset = ConversationDataset(
        root_dir=args.dataset_dir,
        tokenizer=tokenizer,
        model=model,
        config=config,
        max_length=args.max_length,
        device=args.device
    )

    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.custom_collate_fn)
    test(model, tokenizer, test_dataloader)


if __name__ == "__main__":
    args = get_args()
    infer(args)
