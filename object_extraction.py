import re
import json

def extract_objects_with_bounding_boxes(text):
    pattern = r'(\b\w+\b)\s*\[\[(\d{3},\d{3},\d{3},\d{3})\]\]'
    matches = re.findall(pattern, text)
    return matches

# Define the path to the input JSONL file
input_jsonl_file_path = 'bbox_pope_images/outputs.jsonl'

# Read the JSONL file and process each line
processed_objects = []

with open(input_jsonl_file_path, 'r') as file:
    for line in file:
        try:
            json_line = json.loads(line)
            question_id = json_line.get("question_id", "")
            text = json_line.get("text", "")
            objects = extract_objects_with_bounding_boxes(text)
            for obj, bbox in objects:
                processed_objects.append({
                    "question_id": question_id,
                    "object_name": obj.capitalize(),
                    "bounding_box": bbox
                })
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            continue

# Define the path to the output JSONL file
output_jsonl_file_path = 'objects_with_bounding_boxes.jsonl'

# Save the processed objects to a new JSONL file
with open(output_jsonl_file_path, 'w') as output_file:
    for obj in processed_objects:
        json_line = json.dumps(obj)
        output_file.write(json_line + '\n')

print(f"Objects with bounding boxes saved to '{output_jsonl_file_path}'")
