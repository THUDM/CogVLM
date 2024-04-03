import os
import shutil
from random import shuffle

# Define the paths for the datasets
images_path = 'archive'
labels_path = 'labels'

# Create the directory structure after splitting
split_dirs = {
    'train': 'archive_split/train',
    'valid': 'archive_split/valid',
    'test': 'archive_split/test'
}

# Define the allocation ratios
split_ratios = {'train': 0.8, 'valid': 0.05, 'test': 0.15}

# Define whether to include labels
include_labels = False

for split in split_dirs.values():
    os.makedirs(os.path.join(split, 'images'), exist_ok=True)
    if include_labels:
        os.makedirs(os.path.join(split, 'labels'), exist_ok=True)

# Get all file names (assuming labels and images file names match)
file_names = [f.split('.')[0] for f in os.listdir(images_path) if f.endswith('.jpg') or f.endswith('.png')]
shuffle(file_names)  # Randomly shuffle the list of file names


# Calculate the number of files each split should contain
total_files = len(file_names)
split_counts = {split: int(ratio * total_files) for split, ratio in split_ratios.items()}
print(f"Split counts: {split_counts}")

# Allocate files to the corresponding split directories
start = 0
for split, count in split_counts.items():
    end = start + count
    print(f"Processing {split} split with {count} files")
    image_count = 0
    label_count = 0
    for file_name in file_names[start:end]:
        for ext in ['.jpg', '.png']:
            src_file = os.path.join(images_path, f'{file_name}{ext}')
            if os.path.exists(src_file):
                shutil.copy(src_file, os.path.join(split_dirs[split], 'images'))
                image_count += 1
                break
        if include_labels:
            label_file = os.path.join(labels_path, f'{file_name}.json')
            shutil.copy(label_file, os.path.join(split_dirs[split], 'labels'))
            label_count += 1
    print(f"Copied {image_count} image files and {label_count} label files to {split} split")
    start = end