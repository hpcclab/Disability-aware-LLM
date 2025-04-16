'''
This code works only edge except nemoguardrails.
'''
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
import pyttsx3 
from nemoguardrails import LLMRails, RailsConfig
import nest_asyncio
import os

model_id = "meta-llama/Llama-Guard-3-1B"

model_guard = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

local_model_path = snapshot_download(model_id)
print(f"Model downloaded to: {local_model_path}")

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


NVIDIA_API_KEY='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'
nest_asyncio.apply()

os.environ["NVIDIA_API_KEY"]="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"

config = RailsConfig.from_path("./config")

rails = LLMRails(config)
def nemo(text):
  completion = rails.generate(
      messages=[{"role": "user", "content": text}],)
  
  print(completion["content"])
  return completion["content"]

nemo_final=nemo(final_res)

engine = pyttsx3.init()
voices = engine.getProperty('voices')

engine.setProperty('voice', voices[19].id)
engine.setProperty('volume', 1.0)
engine.setProperty('rate', 180)
engine.say(nemo_final)
engine.runAndWait()
