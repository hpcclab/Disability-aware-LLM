import cv2
import os
import time
from ultralytics import YOLO
from collections import Counter

model = YOLO("yolov8s.pt")

image_folder = '../dataset/train'
output_folder = 'output_results'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

log_file = 'detection_log.txt'

with open(log_file, 'w') as log:
    log.write("YOLO Detection Log with Inference Time:\n\n")

    image_files = [entry.name for entry in os.scandir(image_folder) if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = image_files[:8000]

    for img_name in image_files:
        img_path = os.path.join(image_folder, img_name)
        print(f"Processing {img_name}...")
        img = cv2.imread(img_path)

        results = model(img)

        # Accurate inference time from YOLO (in milliseconds)
        inference_time_ms = results[0].speed['inference']
        inference_time_sec = inference_time_ms / 1000.0

        object_names = results[0].names
        detected_classes = results[0].boxes.cls.cpu().numpy()
        detected_objects = [object_names[int(cls)] for cls in detected_classes]
        object_count = Counter(detected_objects)

        result_image = results[0].plot()
        result_image_path = os.path.join(output_folder, f"detected_{img_name}")
        cv2.imwrite(result_image_path, result_image)

        log.write(f"Image: {img_name}\n")
        log.write(f"Inference Time: {inference_time_sec:.4f} seconds\n")
        log.write("Detected objects:\n")
        if object_count:
            for obj, count in object_count.items():
                log.write(f"{obj}: {count}\n" if count > 1 else f"{obj}\n")
        else:
            log.write("No objects detected\n")
        log.write("\n")

print("Processing complete. Inference times and detections are saved in 'detection_log.txt'.")
