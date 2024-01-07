import os
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image


class ConversationDataset(Dataset):
    def __init__(self, root_dir, tokenizer, model, config, max_length=512, device='cuda:0'):
        self.root_dir = root_dir
        self.tokenizer = tokenizer
        self.model = model
        self.config = config
        self.image_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        self.filenames = os.listdir(self.image_dir)
        self.max_length = max_length
        self.device = device
        self.torch_type = torch.bfloat16

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def custom_collate_fn(batch):
        batched_data = {}
        for key in batch[0].keys():
            if isinstance(batch[0][key], list):
                batched_data[key] = [batch_item[key] for batch_item in batch]
            elif isinstance(batch[0][key], torch.Tensor):
                batched_data[key] = torch.stack([item[key] for item in batch])
            else:
                raise ValueError("Unsupported datatype in custom collate_fn")

        return batched_data

    def __getitem__(self, idx):

        img_name = os.path.join(self.image_dir, self.filenames[idx])
        img_basename = os.path.splitext(self.filenames[idx])[0]
        label_name = os.path.join(self.label_dir, f"{img_basename}.json")

        image = Image.open(img_name).convert('RGB')
        with open(label_name, 'r') as f:
            label_data = json.load(f)

        num_rounds = len(label_data["conversations"]) // 2
        sampled_round_id = random.randint(0, num_rounds - 1)
        history = [(label_data["conversations"][(sampled_round_id - 1) * 2]["content"],
                    label_data["conversations"][(sampled_round_id - 1) * 2 + 1]["content"])] if (
                sampled_round_id > 0 and random.random() > 0.5) else None

        query = label_data["conversations"][sampled_round_id * 2]["content"]
        response = label_data["conversations"][sampled_round_id * 2 + 1]["content"]

        input_data = self.model.build_conversation_input_ids(
            tokenizer=self.tokenizer,
            query=query,
            history=history,
            images=[image],
            answer=response,
            template_version="vqa",
        )

        def pad_to_len(unpadded_tensor, pad_to_length, pad_value=0):
            if len(unpadded_tensor) >= pad_to_length:
                return unpadded_tensor[:pad_to_length]
            return torch.cat(
                (unpadded_tensor,
                 torch.full(
                     [pad_to_length - len(unpadded_tensor)],
                     fill_value=pad_value,
                     dtype=unpadded_tensor.dtype,
                     device=unpadded_tensor.device)), dim=0)


        input_data['attention_mask'] = pad_to_len(
            input_data['attention_mask'],
            self.max_length,
            pad_value=0
        )
        input_data['token_type_ids'] = pad_to_len(
            input_data['token_type_ids'],
            self.max_length,
            pad_value=0
        )
        input_data['input_ids'] = pad_to_len(
            input_data['input_ids'],
            self.max_length,
            pad_value=self.tokenizer.pad_token_id
        )
        input_data['labels'] = pad_to_len(
            input_data['labels'],
            self.max_length,
            pad_value=-100
        )

        for data_key in input_data:
            if data_key in ['cross_images', 'images']:
                input_data[data_key] = [data.to(self.device).to(self.torch_type) for data in input_data[data_key]]
            else:
                input_data[data_key] = input_data[data_key].to(self.device)

        return input_data
