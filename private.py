import json
import base64
import os
import time
import pyttsx3
import nest_asyncio
from openai import OpenAI
from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv
load_dotenv()
# Load environment variables   
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY is not set in the environment variables.")


# Initialize Hive AI client
client = OpenAI(
    base_url="https://api.thehive.ai/api/v3/",  # Hive AI's endpoint
    api_key=os.getenv("HIVE_API_KEY")  # Replace with your API key
)

# Set up NeMo Guardrails
nest_asyncio.apply()
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Initialize Text-to-Speech
# engine = pyttsx3.init()
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[19].id)  # Adjust if needed
# engine.setProperty('volume', 1.0)
# engine.setProperty('rate', 180)

def get_completion(prompt, image_path, model="meta-llama/llama-3.2-11b-vision-instruct"):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    start_time = time.time()  # Start response time tracking
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": base64_image}}
            ]}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    
    end_time = time.time()  # End response time tracking
    
    return response.choices[0].message.content, end_time - start_time  # Return AI response and time taken

def nemo(text):
    completion = rails.generate(messages=[{"role": "user", "content": text}])
    return completion["content"]

# Load test.json
test_json_path = "dataset/test/test_1000_updated.json"
test_folder = "dataset/test_1000"  # Folder containing images
output_file = "results.txt"

with open(test_json_path, "r") as file:
    test_data = json.load(file)

with open(output_file, "w") as result_file:
    for item in test_data:
        image_name = item["image"]
        try:
            image_number = int(image_name.split("_")[-1].split(".")[0])  # Extract number from filename
            if not (1 <= image_number <= 1000):  # Process only the specified range
                continue
        except ValueError:
            continue  # Skip if filename is not in expected format

        question = item["question"]
        category = item["category"]  # Directly using the category from JSON
        image_path = os.path.join(test_folder, image_name)

        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        print(f"Processing {image_name} ({category} Query)...")

        total_start_time = time.time()  # Start total processing time

        # Get AI response
        ai_response, response_time = get_completion(question, image_path)

        # Apply NeMo Guardrails
        final_response = nemo(ai_response)

        total_end_time = time.time()  # End total processing time
        total_processing_time = total_end_time - total_start_time

        # Write to output file
        result_file.write(f"Image: {image_name}\n")
        result_file.write(f"Question: {question}\n")
        result_file.write(f"Category: {category}\n")
        result_file.write(f"AI Response: {ai_response}\n")
        result_file.write(f"Final Response: {final_response}\n")
        result_file.write(f"AI Response Time: {response_time:.2f} seconds\n")
        result_file.write(f"Total Processing Time: {total_processing_time:.2f} seconds\n\n")

        print(f"Completed: {image_name} | AI Response Time: {response_time:.2f}s | Total Time: {total_processing_time:.2f}s\n")

print("Processing completed.")
