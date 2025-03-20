import json
import base64
import os
import time
import cv2
import whisper
import pyaudio
import wave
import numpy as np
import nest_asyncio
from openai import OpenAI
from nemoguardrails import LLMRails, RailsConfig

# Initialize Whisper model
whisper_model = whisper.load_model("small")  # Choose "base", "small", "medium", "large" as needed

# Initialize NeMo Guardrails
NVIDIA_API_KEY = "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"
nest_asyncio.apply()
os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

# Initialize Hive AI client
client = OpenAI(
    base_url="https://api.thehive.ai/api/v3/",  # Hive AI's endpoint
    api_key="HwDF5vDdbekdQbWsjcrsAXfsZo53N2v7"  # Replace with your API key
)

def record_audio(filename="query.wav", silence_threshold=500, timeout=3):
    """Records audio until a pause is detected and saves it as a WAV file."""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    print("Listening for query...")
    frames = []
    silent_chunks = 0

    while True:
        data = stream.read(CHUNK)
        np_data = np.frombuffer(data, dtype=np.int16)
        frames.append(data)

        if np.abs(np_data).mean() < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks > (timeout * (RATE / CHUNK)):  # Stop if silence is detected for the timeout duration
            break

    print("Stopped recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    return filename
def transcribe_audio(audio_path):
    """Transcribes speech from an audio file using OpenAI's Whisper."""
    if not audio_path or not os.path.exists(audio_path):
        return "", "", None  # Return empty strings if audio is missing

    print("Transcribing audio...")

    # Transcribe using Whisper
    result = whisper_model.transcribe(audio_path, language="en")

    transcribed_text = result["text"]
    detected_language = result["language"]

    print(f"Transcribed Text: {transcribed_text}")
    print(f"Detected Language: {detected_language}")

    return transcribed_text

# def transcribe_audio(audio_path):
#     """Transcribes recorded audio into text using Whisper."""
#     print("Transcribing audio...")
#     result = whisper_model.transcribe(audio_path)
#     return result["text"]

def capture_image():
    """Captures an image from the webcam and saves it."""
    cap = cv2.VideoCapture(0)  # Open the default camera
    time.sleep(1)  # Give the camera time to adjust
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture image")
        cap.release()
        return None
    
    image_path = "captured_image.jpg"
    cv2.imwrite(image_path, frame)  # Save the captured image
    cap.release()
    print("Image Captured.")
    return image_path

def get_completion(prompt, image_path, model="meta-llama/llama-3.2-11b-vision-instruct"):
    """Sends an image and text prompt to the AI model."""
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
    """Processes AI response through NeMo Guardrails."""
    completion = rails.generate(messages=[{"role": "user", "content": text}])
    return completion["content"]

# Step 1: Capture voice input
audio_file = record_audio()

# Step 2: Convert voice to text
query = transcribe_audio(audio_file).strip()

if query:
    print(f"Query Recognized: {query}")

    # Step 3: Capture an image after detecting silence
    image_path = capture_image()

    if image_path:
        print(f"Processing Captured Image Query...")

        total_start_time = time.time()  # Start total processing time

        # Step 4: Get AI response
        ai_response, response_time = get_completion(query, image_path)

        # Step 5: Apply NeMo Guardrails
        final_response = nemo(ai_response)

        total_end_time = time.time()  # End total processing time
        total_processing_time = total_end_time - total_start_time

        # Display results
        print("\n--- RESULTS ---")
        print(f"Query: {query}")
        print(f"AI Response: {ai_response}")
        print(f"Final Response: {final_response}")
        print(f"AI Response Time: {response_time:.2f} seconds")
        print(f"Total Processing Time: {total_processing_time:.2f} seconds")
    else:
        print("No image captured. Exiting.")
else:
    print("No query detected. Exiting.")
