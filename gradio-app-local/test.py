import torch
from transformers import pipeline
import os

offload_folder = os.getcwd()
local_model_path = "./local_model"
if not os.path.exists(local_model_path):
    os.makedirs(local_model_path)
pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b-jpn-it",
    model_kwargs={"torch_dtype": torch.float16},
    device="cuda",  # replace with "mps" to run on a Mac device
    # device_map="auto",
)

messages = [
    {"role": "user", "content": "クリスマスキャロルについてあらすじを教えてください。"},
]

print("************************")
outputs = pipe(messages, return_full_text=False, max_new_tokens=256)
assistant_response = outputs[0]["generated_text"].strip()
print(assistant_response)