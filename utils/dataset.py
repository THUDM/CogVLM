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
    print_rank0(f'Found {len(target_files)} files...')
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
        all_files = find_all_files(data_dir, suffix=".jpg")
        print_rank0(f"Found {len(all_files)} samples in total...")
        return all_files
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        # img
        try:
            img = Image.open(data).convert('RGB')
        except Exception as e:
            print_rank0(f"Image loading failed for {data} with error: {e}", level=logging.WARNING)
            return {}

        img_dict = self.process_img(img)
        
        # text
        try:
            label = data.split('/')[-2]  # Extracting directory name as label
            uni_key = label
            text_dict = self.process_text(label, "Diagnosis:")
        except Exception as e:
            print_rank0(f"Text processing failed for {data} with error: {e}", level=logging.WARNING)
            return {}

        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {data}", level=logging.WARNING)
            return {}

        # other attr
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        return ret
