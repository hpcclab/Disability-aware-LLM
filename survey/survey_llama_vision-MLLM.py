import ollama
import time
import json
import os
from pathlib import Path

# Initialize variables
model_name = "llama3.2-vision:11b"

# File paths
test_json_path = "../dataset/test.json"
test_folder = "../dataset/test"
output_file = "results_vision.txt"
progress_file = "progress.json"

def safe_json_load(filepath):
    """Safely load JSON file with error handling"""
    try:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            with open(filepath, "r") as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Could not load {filepath} - {str(e)}")
    return {}

def safe_json_save(data, filepath):
    """Safely save data to JSON file"""
    try:
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except IOError as e:
        print(f"Error: Could not save {filepath} - {str(e)}")
        return False

# Load test data
with open(test_json_path, "r") as file:
    test_data = json.load(file)

# Load progress if exists
progress = safe_json_load(progress_file)
processed_images = progress.get("processed_images", [])
interrupted_image = progress.get("interrupted_image", None)

# Determine where to start processing
start_index = 0
if interrupted_image:
    try:
        start_index = [item["image"] for item in test_data].index(interrupted_image)
        print(f"\nFound interruption point at image: {interrupted_image}")
        resume = input("Resume from this image? (y/n): ").lower() == 'y'
        if not resume:
            processed_images = []
            interrupted_image = None
            start_index = 0
    except ValueError:
        print(f"\nWarning: Interrupted image {interrupted_image} not found in test data")
        processed_images = []
        interrupted_image = None

results = []
print("\nStarting Llama Survey with test data...")

try:
    with open(output_file, "a" if processed_images else "w") as result_file:
        for i in range(start_index, len(test_data)):
            item = test_data[i]
            image_name = item["image"]
            question = item["question"]
            
            # Skip if already successfully processed (unless it's our resume point)
            if image_name in processed_images and image_name != interrupted_image:
                continue
                
            image_path = os.path.join(test_folder, image_name)
            
            if not os.path.exists(image_path):
                print(f"\nImage not found: {image_path}")
                continue
            
            print(f"\nProcessing image: {image_name}")
            print(f"Question: {question}")
            
            # Timing variables
            start_time = time.perf_counter()
            first_chunk_received = False
            first_chunk_time = None
            full_response = []
            
            print("Response (streaming): ", end="", flush=True)
            
            # Stream the response
            stream = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": question,
                        'images': [image_path]
                    }
                ],
                stream=True
            )
            
            for chunk in stream:
                if not first_chunk_received:
                    first_chunk_time = time.perf_counter() - start_time
                    first_chunk_received = True
                    
                content = chunk['message']['content']
                print(content, end="", flush=True)
                full_response.append(content)
            
            response_text = ''.join(full_response)
            completion_time = time.perf_counter() - start_time
            
            # Save results
            result = {
                "question": question,
                "response": response_text,
                "image": image_name,
                "first_chunk_time": first_chunk_time,
                "completion_time": completion_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            results.append(result)
            
            # Write to output file
            result_file.write(f"Question: {question}\n")
            result_file.write(f"Image: {image_name}\n")
            result_file.write(f"Response: {response_text}\n")
            result_file.write(f"First Chunk Time: {first_chunk_time:.4f} seconds\n")
            result_file.write(f"Completion Time: {completion_time:.4f} seconds\n")
            result_file.write("-" * 50 + "\n")
            
            print(f"\nFirst Chunk Time: {first_chunk_time:.4f} seconds")
            print(f"Completion Time: {completion_time:.4f} seconds")
            print("-" * 50)
            
            # Update progress (only add to processed_images after successful completion)
            if image_name not in processed_images:
                processed_images.append(image_name)
            
            # Clear interrupted_image flag once we've successfully processed it
            if image_name == interrupted_image:
                interrupted_image = None
            
            # Save progress for next image
            progress_data = {
                "processed_images": processed_images,
                "interrupted_image": test_data[i+1]["image"] if i+1 < len(test_data) else None
            }
            if not safe_json_save(progress_data, progress_file):
                print("Warning: Failed to save progress file")

except KeyboardInterrupt:
    print("\nProcess interrupted by user. Saving progress...")
    # Save the current image as interruption point
    progress_data = {
        "processed_images": processed_images,
        "interrupted_image": image_name
    }
    safe_json_save(progress_data, progress_file)
except Exception as e:
    print(f"\nAn error occurred: {str(e)}")
    # Save the current image as interruption point
    progress_data = {
        "processed_images": processed_images,
        "interrupted_image": image_name
    }
    safe_json_save(progress_data, progress_file)