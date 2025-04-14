import socket
import os
import warnings
import time
import numpy as np
import re
import torch
import pyttsx3
import nest_asyncio
import nltk
from nltk import sent_tokenize
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
import ollama
from gtts import gTTS
from PIL import Image
import whisper
import gradio as gr
from nemoguardrails import LLMRails, RailsConfig

warnings.filterwarnings("ignore")
nltk.download("punkt")

# -------------------------
# Constants
EDGE_HOSTNAME = "edge-device-01"  # Replace this with your actual edge device hostname
MODEL_ID = "meta-llama/Llama-Guard-3-1B"
LOCAL_MODEL_PATH = snapshot_download(MODEL_ID)
os.environ["NVIDIA_API_KEY"] = "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"
nest_asyncio.apply()

# Load Guard Model
model_guard = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# -------------------------
# Run the safety check
input_text = "What the person is doing?"

conversation = [
    {
        "role": "user",
        "content": [{"type": "text", "text": input_text}],
    }
]

input_ids = tokenizer.apply_chat_template(
    conversation, return_tensors="pt"
).to(model_guard.device)

prompt_len = input_ids.shape[1]

output = model_guard.generate(
    input_ids,
    max_new_tokens=100,
    temperature=1,
    pad_token_id=0,
)

generated_tokens = output[:, prompt_len:]
result = tokenizer.decode(generated_tokens[0])
print(f"Guard Result: {result}")

# -------------------------
# If safe and on edge device, run LLaVA
if result.strip() == "safe<|eot_id|>" and socket.gethostname() == EDGE_HOSTNAME:
    print("Running on edge device. Invoking LLaVA...")
    res = ollama.chat(
        model="llava:7b",
        messages=[
            {
                'role': 'user',
                'content': input_text,
                'images': ['person.jpeg']
            }
        ]
    )
    final_res = res['message']['content']
else:
    final_res = "The question is not appropriate or not running on edge device."
print(f"LLaVA Result: {final_res}")

# -------------------------
# Run NeMo Guardrails
config = RailsConfig.from_path("./config")
rails = LLMRails(config)

def nemo(text):
    completion = rails.generate(
        messages=[{"role": "user", "content": text}],
    )
    print(f"NeMo Output: {completion['content']}")
    return completion["content"]

nemo_final = nemo(final_res)

# -------------------------
# Text-to-Speech
engine = pyttsx3.init()
voices = engine.getProperty('voices')
if len(voices) > 19:
    engine.setProperty('voice', voices[19].id)

engine.setProperty('volume', 1.0)
engine.setProperty('rate', 180)
engine.say(nemo_final)
engine.runAndWait()

