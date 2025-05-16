import json
import re

input_json_path = "filtered_train_data.json"           
yolo_log_path = "detection_log.txt"         
output_json_path = "enriched_data.json"


with open(input_json_path, 'r') as f:
    input_data = json.load(f)


yolo_data = {}
with open(yolo_log_path, 'r') as f:
    content = f.read()

# Split log by image entries
entries = content.strip().split("Image:")

for entry in entries:
    if not entry.strip():
        continue

    lines = entry.strip().split("\n")
    image_line = lines[0].strip()
    image_name = image_line

    # Find inference time
    inference_time_match = re.search(r"Inference Time:\s*([0-9.]+)", entry)
    inference_time = float(inference_time_match.group(1)) if inference_time_match else None

    # Get detected objects
    detected_objects = []
    for line in lines[2:]:
        obj_line = line.strip()
        print(obj_line)
        # Skip if line is empty, 'Detected objects:', or 'No objects detected'
        if not obj_line or obj_line.lower() == "no objects detected" or obj_line.lower() == "detected objects:":
            continue

        # Remove colon and count if present (e.g., 'book: 2')
        obj = re.sub(r":\s*\d+", "", obj_line)
        detected_objects.append(obj)

    yolo_data[image_name] = {
        "inference_time": inference_time,
        "detected_objects": detected_objects
    }


for item in input_data:
    image_name = item["image"]
    detection = yolo_data.get(image_name, {"inference_time": None, "detected_objects": []})
    item["inference_time"] = detection["inference_time"]
    item["detected_objects"] = detection["detected_objects"]


with open(output_json_path, 'w') as f:
    json.dump(input_data, f, indent=2)

print(f"✅ Enriched JSON saved to '{output_json_path}'")
