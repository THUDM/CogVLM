import os
import logging
import random
import logging
import jsonlines
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0

def find_all_files(path, suffix=".jpg"):
    target_files = []
    for cur_dir, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith(suffix):
                target_files.append(os.path.join(cur_dir, f))
    print_rank0(f'find {len(target_files)} files...')
    return target_files

class ItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, **kwargs):
        super().__init__()
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor = image_processor, text_processor
    
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        return img_dict
    
    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)
    
    def load_data(self, data_dir):
        data_labels = []
        for label_dir in os.listdir(data_dir):
            full_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(full_path):
                for img_file in os.listdir(full_path):
                    if img_file.endswith(".jpg"):
                        img_path = os.path.join(full_path, img_file)
                        data_labels.append((img_path, label_dir))
        print_rank0(f"find {len(data_labels)} samples in all...")
        return data_labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        # img
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        img_dict = self.process_img(img)
        # text
        text_dict = self.process_text(label)
        if text_dict is None:
            print_rank0(f"Process text failed.\n The data is {img_path}", level=logging.WARNING)
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": os.path.basename(img_path)}
        return ret