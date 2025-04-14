import socket
import os
import warnings
import torch
import nest_asyncio
import pyttsx3
import speech_recognition as sr
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import ollama
from nemoguardrails import LLMRails, RailsConfig

warnings.filterwarnings("ignore")
nest_asyncio.apply()

EDGE_HOSTNAME = "edge-device-01"
MODEL_ID = "meta-llama/Llama-Guard-3-1B"
LOCAL_MODEL_PATH = snapshot_download(MODEL_ID)
os.environ["NVIDIA_API_KEY"] = "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"

model_guard = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def recognize_user_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Please say your name for identification...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        user_name = recognizer.recognize_google(audio)
        print(f"User identified: {user_name}")
        return user_name
    except sr.UnknownValueError:
        print("Could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None

def is_running_on_edge():
    return socket.gethostname() == EDGE_HOSTNAME

def run_voice_identification_and_process(input_text):
    if not is_running_on_edge():
        print("This program must be run on an edge device.")
        return

    user_name = recognize_user_voice()
    if user_name is None:
        print("Failed to identify the user via voice.")
        return

    conversation = [
        {
            "role": "user",
            "content": [{"type": "text", "text": input_text}],
        }
    ]
    
    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model_guard.device)
    prompt_len = input_ids.shape[1]

    output = model_guard.generate(input_ids, max_new_tokens=100, temperature=1, pad_token_id=0)
    generated_tokens = output[:, prompt_len:]
    result = tokenizer.decode(generated_tokens[0])
    print(f"Guard Result: {result}")

    if result.strip() == "safe<|eot_id|>" and is_running_on_edge():
        print("Running on edge device. Invoking LLaVA...")
        res = ollama.chat(
            model="llava:7b",
            messages=[{
                'role': 'user',
                'content': input_text,
                'images': ['person.jpeg']
            }]
        )
        final_res = res['message']['content']
    else:
        final_res = "The question is not appropriate or not running on edge device."
    
    print(f"LLaVA Result: {final_res}")

    config = RailsConfig.from_path("./config")
    rails = LLMRails(config)

    def nemo(text):
        completion = rails.generate(
            messages=[{"role": "user", "content": text}],
        )
        print(f"NeMo Output: {completion['content']}")
        return completion["content"]

    nemo_final = nemo(final_res)

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    if len(voices) > 19:
        engine.setProperty('voice', voices[19].id)

    engine.setProperty('volume', 1.0)
    engine.setProperty('rate', 180)
    engine.say(f"Response for {user_name}: {nemo_final}")
    engine.runAndWait()

    with open("user_interactions.log", "a") as log_file:
        log_file.write(f"User: {user_name}, Query: {input_text}, Response: {nemo_final}\n")

input_text = "What is the person doing?"

run_voice_identification_and_process(input_text)
