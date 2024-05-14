import fiftyone as fo
import fiftyone.zoo as foz
import json 
    
pope_img_names = [json.loads(q) for q in open("coco_pope_random.json", 'r')]
image_names = [line["image"] for line in pope_img_names]
image_names = set(image_names)

coco_val_segmentations = [json.loads(q) for q in open("coco_ground_truth_segmentation.json", 'r')]
image_ids = [line["image_id"] for line in coco_val_segmentations if line["image"] in image_names]
dataset = foz.load_zoo_dataset(
    "coco-2014",
    split="validation",
    image_ids = image_ids
)
session = fo.launch_app(dataset,remote=True)