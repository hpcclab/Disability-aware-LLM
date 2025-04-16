'''
This code works across edge and cloud.
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
from openai import OpenAI
import pyttsx3 
from nemoguardrails import LLMRails, RailsConfig
import nest_asyncio
import os

model_id = "meta-llama/Llama-Guard-3-1B"

local_model_path='/Users/bhanu/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-1B/snapshots/acf7aafa60f0410f8f42b1fa35e077d705892029'

model_guard = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

client = OpenAI(
    base_url="https://api.thehive.ai/api/v3/",
    api_key="HwDF5vDdbekdQbWsjcrsAXfsZo53N2v7"
)

def get_completion(prompt, model = "meta-llama/llama-3.2-11b-vision-instruct"):
  response = client.chat.completions.create(
    model=model,
    messages=[
      {
        "role": "user",
        "content": [
          {"type": "text", "text": prompt},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://d24edro6ichpbm.thehive.ai/example-images/vlm-example-image.jpeg"
            }
          }
        ]
      }
    ],
    temperature=0.7,
    max_tokens=1000
  )

  return response.choices[0].message.content

final_res=get_completion("What's in this image?")

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
