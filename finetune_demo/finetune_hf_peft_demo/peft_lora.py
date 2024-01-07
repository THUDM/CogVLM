import argparse

from tqdm import tqdm
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import json
import os
import random
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer, get_linear_schedule_with_warmup


class ConversationDataset(Dataset):
    def __init__(self, root_dir, tokenizer, model, config, device='cuda:0'):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.filenames = os.listdir(self.image_dir)
        self.output_length = 800
        self.input_length = 800
        self.device = device
        self.torch_type = torch.bfloat16

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def custom_collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            # For list[tensor]] structures
            if isinstance(batch[0][key], list):
                batched_data[key] = [batch_item[key] for batch_item in batch]
            # For tensor structures
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                raise ValueError("Unsupported datatype in custom collate_fn")

        return batched_data

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.filenames[idx])
        label_name = os.path.join(self.label_dir, self.filenames[idx].replace('.jpg', '.json'))

        image = Image.open(img_name).convert('RGB')
        with open(label_name, 'r') as f:
            label_data = json.load(f)

        num_rounds = len(label_data["conversations"]) // 2
        # sampled round to train

        sampled_round_id = random.randint(0, num_rounds - 1)
        # sampled_rounds-1 -th rounds are used as history

        history = [(label_data["conversations"][(sampled_round_id - 1) * 2]["content"],
                    label_data["conversations"][(sampled_round_id - 1) * 2 + 1]["content"])] if (
                sampled_round_id > 0 and random.random() > 0.5) else None

        # the last sampled round is used as query & response
        query = label_data["conversations"][sampled_round_id * 2]["content"]
        response = label_data["conversations"][sampled_round_id * 2 + 1]["content"]

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            history=history,
            images=[image],
            answer=response
        )

        def pad_to_len(unpadded_tensor, pad_to_length, pad_value=0):
            if len(unpadded_tensor) >= pad_to_length:
                return unpadded_tensor[:pad_to_length]
            return torch.cat(
                (unpadded_tensor, torch.full([pad_to_length - len(unpadded_tensor)],
                                             fill_value=pad_value,
                                             dtype=unpadded_tensor.dtype,
                                             device=unpadded_tensor.device)), dim=0)

        input_data['attention_mask'] = pad_to_len(
            input_data['attention_mask'],
            self.output_length,
            pad_value=0
        )
        input_data['token_type_ids'] = pad_to_len(
            input_data['token_type_ids'],
            self.output_length,
            pad_value=0
        )
        input_data['input_ids'] = pad_to_len(
            input_data['input_ids'],
            self.output_length,
            pad_value=self.tokenizer.pad_token_id
        )
        input_data['labels'] = pad_to_len(
            input_data['labels'],
            self.output_length,
            pad_value=-100
        )

        for data_key in input_data:
            if data_key in ['cross_images', 'images']:
                input_data[data_key] = [data.to(self.device).to(self.torch_type) for data in input_data[data_key]]
            else:
                input_data[data_key] = input_data[data_key].to(self.device)

        return input_data


def get_args():
    parser = argparse.ArgumentParser()

    # learning parameters
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)

    # model and dataset parameters
    parser.add_argument("--model_dir", type=str, default="/share/hwy/code/CogVLM_release/checkpoints/cogagent-chat-hf")
    parser.add_argument("--tokenizer_dir", type=str, default="/share/official_pretrains/hf_home/vicuna-7b-v1.5")
    parser.add_argument("--dataset_dir", type=str, default="/share/img_datasets/zyx_finetune_dataset")
    parser.add_argument("--train_rate", type=float, default=0.8)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--save_dir", type=str, default="output")

    # lora_parameters
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--lora_alpha", type=int, default=32)

    return parser.parse_args()


def finetune(args):
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                 trust_remote_code=True).cuda()

    # Instantiate your Dataset
    dataset = ConversationDataset(root_dir=args.dataset_dir,
                                  tokenizer=tokenizer,
                                  model=model,
                                  config=config,
                                  device=args.device
                                  )

    train_size = int(args.train_rate * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  collate_fn=dataset.custom_collate_fn)
    eval_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.custom_collate_fn)

    peft_config = LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,
                             inference_mode=False,
                             r=args.lora_rank,
                             target_modules=
                             [
                                 "vision_expert_query_key_value",  # vision expert's attention
                                 "q_proj", "k_proj", "v_proj",  # cross vit
                                 "query_key_value",  # vit
                                 "language_expert_query_key_value"  # language's attention
                             ],
                             lora_alpha=args.lora_rank,
                             lora_dropout=args.lora_dropout
                             )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # start training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            outputs = model(**batch)

            loss = outputs.loss
            print("loss: ", loss)
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for step, batch in enumerate(tqdm(eval_dataloader)):
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                                       skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(args.save_dir, f"epoch_{epoch + 1}")
            model.save_pretrained(save_path)
            print(f"Model saved to {save_path}")

    save_path = os.path.join(args.save_dir, f"epoch_last")
    model.save_pretrained(save_path)


if __name__ == "__main__":
    args = get_args()
    finetune(args)
