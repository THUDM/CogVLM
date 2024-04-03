import os
import logging
import random
import json
from io import BytesIO
from PIL import Image
from torch.utils.data import Dataset
from sat.helpers import print_rank0

class ItemDataset(Dataset):
    # Initialization function, set image processor, text processor, data directories, etc.
    def __init__(self, image_processor, text_processor, args, data_dirs, cross_image_processor=None, **kwargs):
        super().__init__()
        self.image_processor, self.text_processor, self.cross_image_processor = image_processor, text_processor, cross_image_processor
        self.data_dirs = data_dirs
        self.data = self.load_data(data_dirs)
        print_rank0(f"Dataset initialized with {len(self.data)} samples.")

    # Define a function to find all target files
    def find_all_files(self, path, suffixes=(".jpg", ".png")):
        target_files = []
        for cur_dir, _, files in os.walk(path, followlinks=True):
            for f in files:
                if f.endswith(suffixes):
                    target_files.append(os.path.join(cur_dir, f))
        print_rank0(f'Found {len(target_files)} files...')
        return target_files
    
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
        image_files = sorted(self.find_all_files(image_dir, suffixes=(".jpg", ".png")))

        # Check if label directory exists
        if os.path.exists(label_dir):
            # Construct label file paths based on image file names
            label_files = [os.path.join(label_dir, os.path.splitext(os.path.basename(image_file))[0] + '.json') for image_file in image_files]
            # Check if all label files exist
            for label_file in label_files:
                if not os.path.exists(label_file):
                    raise FileNotFoundError(f"Label file {label_file} does not exist.")
        else:
            # If label directory does not exist, use image file names as labels
            label_files = image_files
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
            if os.path.exists(label_file):
                with open(label_file, 'r') as f:
                    label_data = json.load(f)
                    label = label_data['captions'][0]['content']
            else:
                label = os.path.splitext(os.path.basename(image_file))[0]
        except Exception as e:
            print_rank0(e, level=logging.WARNING)
            return {}
        uni_key = label
        prompt = "CAPTCHA:" if not os.path.exists(label_file) else "This image can be best described as:"
        text_dict = self.process_text(label, prompt)
        if text_dict is None:
            print_rank0(f"Process text failed. Please check the max_target_length & max_source_length.\n The data is {label_file}", level=logging.WARNING)
            return {}
        # other attr
        ret = {**img_dict, **text_dict, "question_id": uni_key}
        return ret