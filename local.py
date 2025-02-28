# ******** Code that helps to download the LLama Guard ******


# model_guard = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
# )

# local_model_path = snapshot_download(model_id)
# print(f"Model downloaded to: {local_model_path}")
# ***************

# code for interacting with llava 7b 
# **************


# res = ollama.chat(
# 	model="llava:7b",
# 	messages=[
# 		{
# 			'role': 'user',
# 			'content': 'Describe this image:',
# 			'images': ['image.jpg']
# 		}
# 	]
# )

# print(res['message']['content'])
# **************


# *******Tried downloading the llava - which actually occupies nearly 16GB. ******

# from huggingface_hub import snapshot_download

# model_id = "llava-hf/llava-1.5-7b-hf"

# # Download and store the model locally
# local_model_path = snapshot_download(model_id)

# print(f"Model is downloaded and cached at: {local_model_path}")

# *******************



import whisper
import gradio as gr
import time
import warnings
import os
from gtts import gTTS
from PIL import Image
import nltk
from nltk import sent_tokenize
warnings.filterwarnings("ignore")
import numpy as np
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from huggingface_hub import snapshot_download
import ollama

model_id = "meta-llama/Llama-Guard-3-1B"

local_model_path='/Users/bhanu/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-1B/snapshots/acf7aafa60f0410f8f42b1fa35e077d705892029'

model_guard = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype="auto",  # Use appropriate dtype
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# **************



# ***********SAFE GUARD ****
input='What the person is doing?'
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": input
            },
        ],
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

result=tokenizer.decode(generated_tokens[0])
print(result)

if result.strip()=="safe<|eot_id|>":

    res = ollama.chat(
        model="llava:7b",
        messages=[
            {
                'role': 'user',
                'content': input,
                'images': ['person.jpeg']
            }
        ]
    )

    final_res=res['message']['content']
    print(final_res)
else:
    final_res="The question is not appropirate to answer"
    print(final_res)


import pyttsx3 

engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Set a specific voice if needed (e.g., first available voice)
engine.setProperty('voice', voices[19].id)

# Adjust volume
engine.setProperty('volume', 1.0)

# Adjust speaking rate
engine.setProperty('rate', 150)
# Dictionary to store object counts

engine.say(final_res)
engine.runAndWait()