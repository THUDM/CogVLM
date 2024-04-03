import os
import logging
import random
import logging
import jsonlines
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0

# Define a function to find all target files
def find_all_files(path, suffixes=(".jpg", ".png")):
    target_files = []
    for cur_dir, _, files in os.walk(path, followlinks=True):
        for f in files:
            if f.endswith(suffixes):
                target_files.append(os.path.join(cur_dir, f))
    print_rank0(f'Found {len(target_files)} files...')
    return target_files

class ItemDataset(Dataset):
    # Initialization function, set image processor, text processor, data directories, etc.
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
        super().__init__()
        self.data = self.load_data(data_dirs)
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
        self.data_dirs = data_dirs
        self.data = self.load_data()
        print_rank0(f"Dataset initialized with {len(self.data)} samples.")
    
    # Function to process images
    def process_img(self, img):
        img_dict = {'vision': self.image_processor(img)}
        if self.cross_image_processor:
            img_dict.update({'cross': self.cross_image_processor(img)})
        print_rank0(f"Image processed.")
        return img_dict
    
    # Function to process text
    def process_text(self, answer, prompt):
        processed_text = self.text_processor(answer, prompt)
        print_rank0(f"Text processed.")
        return processed_text
    
    # Function to load data
    def load_data(self, data_dirs):
        image_dir = os.path.join(self.data_dirs, 'images')
        label_dir = os.path.join(self.data_dirs, 'labels')
        image_files = self.find_all_files(image_dir, suffixes=(".jpg", ".png"))
        label_files = self.find_all_files(label_dir, suffix=".json")
        print_rank0(f"Found {len(image_files)} images and {len(label_files)} labels in total...")
        return list(zip(image_files, label_files))
    
    # Function to get the length of the dataset
    def __len__(self):
        return len(self.data)

    # Function to get an item from the dataset
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