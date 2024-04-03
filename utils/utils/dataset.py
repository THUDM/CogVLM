import os
import logging
import random
import logging
import jsonlines
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0

def find_all_files(path, suffixes=(".jpg", ".png")):
    target_files = []
    for cur_dir, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith(suffixes):
                target_files.append(os.path.join(cur_dir, f))
    print_rank0(f'find {len(target_files)} files...')
    return target_files

class ItemDataset(Dataset):
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
        super().__init__()
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
    
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        return img_dict
    
    def process_text(self, answer, prompt):
        return self.text_processor(answer, prompt)
    
    def load_data(self, data_dirs):
        image_dir = os.path.join(self.data_dirs, 'images')
        label_dir = os.path.join(self.data_dirs, 'labels')
        image_files = self.find_all_files(image_dir, suffixes=(".jpg", ".png"))
        label_files = self.find_all_files(label_dir, suffix=".json")
        print_rank0(f"find {len(image_files)} images and {len(label_files)} labels in all...")
        return list(zip(image_files, label_files))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_file, label_file = self.data[index]
        # img
        try:
            img = Image.open(image_file).convert('RGB')
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        img_dict = self.process_img(img)
        # text
        try:
            with open(label_file, 'r') as f:
                label_data = json.load(f)
                label = label_data['captions'][0]['content']
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        uni_key = label
        text_dict = self.process_text(label, "This image can be best described as:")
        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {label_file}", level=logging.WARNING)
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        return ret