import json
import argparse
from sentence_transformers import SentenceTransformer, util
import jsonlines

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Match objects in images to COCO classes using semantic similarity and output results.")
parser.add_argument("--jsonl_file", type=str, required=True, help="Path to the JSONL file containing image object data.")
parser.add_argument("--output_file", type=str, default="matched.jsonl",help="Path to output the modified JSONL data.")
args = parser.parse_args()

# COCO classes (simplified example, replace with full list as needed)
coco_classes = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", 
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", 
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
]

# Initialize the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def find_coco_matches(jsonl_file, output_file, coco_classes):
    coco_embeddings = model.encode(coco_classes, convert_to_tensor=True)

    with jsonlines.open(jsonl_file) as reader, jsonlines.open(output_file, mode='w') as writer:
        for obj in reader:
            object_name = obj['object_name']
            object_embedding = model.encode(object_name, convert_to_tensor=True)
            cosine_scores = util.cos_sim(object_embedding, coco_embeddings)

            # Find the highest scoring COCO class
            max_score_index = cosine_scores.argmax()
            matched_class = coco_classes[max_score_index]

            # Print and write the output with the matched class
            print(f"Image ID: {obj['question_id']}, Object: {object_name}, Matched COCO Class: {matched_class}, Score: {cosine_scores[0, max_score_index].item()}")
            obj['class'] = matched_class
            writer.write(obj)

# Process the JSONL file to find matches and write results
find_coco_matches(args.jsonl_file, args.output_file, coco_classes)