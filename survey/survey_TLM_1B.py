import ollama
import time
import json
from pathlib import Path
import os

# Initialize variables
model_name = "llama3.2:1b"

# File paths
test_json_path = "enriched_data.json"
output_file = "results_TMLLM:1B.txt"

# Load test data
with open(test_json_path, "r") as file:
    test_data = json.load(file)

results = []

# Run survey with test data
print("\nStarting Llama Survey with test data...")
with open(output_file, "w") as result_file:
    for item in test_data:
        image_name = item["image"]
        question = item["question"]
        detected_objects = item.get("detected_objects", "")
         # Skip if no objects detected or detected_objects contains "None"
        if not detected_objects or len(detected_objects) == 0:
            print(f"Skipping image {image_name} due to no detected objects.")
            continue

        # Format the question with the prompt and detected objects
        prompt = f""" 
        Given the question: \"{question}\", and knowing that the detected objects are: {detected_objects}, please provide an answer solely based on this information, without making any assumptions beyond the provided context.
        """
        
        print(f"Question: {prompt}")
        
        # Start timing before sending the request
        start_time = time.perf_counter()  # Start time for the entire response
        # Streaming metrics
        first_chunk_time = None
        full_response = []
        
        # Track streaming stats
        chunk_count = 0
        total_words = 0
        chunk_times = []
        prev_chunk_time = start_time
        # Start streaming
        print("Response (streaming): ", end="", flush=True)
        response = ollama.chat(model=model_name, messages=[
            {
                "role": "user",
                "content": prompt
            }
        ], stream=True)  # Enable streaming

        # Capture the time for the first chunk (first response received)
        first_chunk_time = None
        response_text = ""
        
        # Streaming loop
        for chunk in response:
            current_time = time.perf_counter()
            if first_chunk_time is None:
                first_chunk_time = time.perf_counter() - start_time  # Time before the first chunk
             # Get content and count words
            content = chunk['message']['content']
            words_in_chunk = len(content.split())
            total_words += words_in_chunk
            
            # Track time between chunks
            time_since_last_chunk = current_time - prev_chunk_time
            chunk_times.append(time_since_last_chunk)
            prev_chunk_time = current_time


             # Print streaming output
            print(content, end="", flush=True)
            full_response.append(content)
            chunk_count += 1

        response_text = ''.join(full_response)  # Append the chunk's content

        # Calculate completion time (total time for the full response)
        completion_time = time.perf_counter() - start_time
          # Calculate streaming metrics
        avg_words_per_chunk = total_words / chunk_count if chunk_count > 0 else 0
        avg_time_between_chunks = sum(chunk_times) / len(chunk_times) if chunk_times else 0
        avg_words_per_second = total_words / completion_time if completion_time > 0 else 0
        # Save result
        result = {
            "question": question,
            "response": response_text,
            "image": image_name,
            "first_chunk_time": first_chunk_time,
            "completion_time": completion_time,
            "streaming_metrics": {
                "avg_words_per_chunk": avg_words_per_chunk,
                "avg_time_between_chunks": avg_time_between_chunks,
                "avg_words_per_second": avg_words_per_second,
                "total_chunks": chunk_count,
                "total_words": total_words,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        results.append(result)
        
       # Write to output file
        result_file.write(f"Question: {question}\n")
        result_file.write(f"Image: {image_name}\n")
        result_file.write(f"Response: {response_text}\n")
        result_file.write(f"First Chunk Time: {first_chunk_time:.4f} seconds\n")
        result_file.write(f"Completion Time: {completion_time:.4f} seconds\n")
        result_file.write(f"Avg Words/Chunk: {avg_words_per_chunk:.2f}\n")
        result_file.write(f"Avg Time Between Chunks: {avg_time_between_chunks:.4f} seconds\n")
        result_file.write(f"Avg Words/Second: {avg_words_per_second:.2f}\n")
        result_file.write("-" * 50 + "\n")
        
        print(f"\nFirst Chunk Time: {first_chunk_time:.4f} seconds")
        print(f"Completion Time: {completion_time:.4f} seconds")
        print(f"Avg Words/Chunk: {avg_words_per_chunk:.2f}")
        print(f"Avg Time Between Chunks: {avg_time_between_chunks:.4f} sec")
        print(f"Avg Words/Second: {avg_words_per_second:.2f}")
        print("-" * 50)
# Save detailed results in JSON format
timestamp = time.strftime("%Y%m%d_%H%M%S")
json_output_file = f"llama_test_results_{timestamp}.json"

with open(json_output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to {json_output_file}")
print(f"Summary results saved to {output_file}")
