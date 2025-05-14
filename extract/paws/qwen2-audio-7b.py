import torch
import os
import pandas as pd
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

folder_name = "qwen2-audio-7b"

def hook_fn(m, i, o, layer_id):
    save_dir = f"../../../activations/paws/{folder_name}/{text_id}/{var_id}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{layer_id}.pt")
    torch.save(o[0][0, -1, :].detach().cpu(), save_path)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B",
    cache_dir=f'../../../huggingface/{folder_name}/',
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B",
    cache_dir=f"../../../huggingface/{folder_name}",
    trust_remote_code=True
)

for i, layer in enumerate(model.language_model.model.layers):
    layer.register_forward_hook(
        lambda m, i, o, layer_id=i: hook_fn(m, i, o, layer_id=layer_id)
    )

df = pd.read_parquet('../../../input/paws/train.parquet')
for text_id in range (10): # Change Test Case
    for var_id in range (1, 3):
        inputs = processor(text=df.loc[text_id, f'sentence{var_id}'], return_tensors="pt")

        with torch.no_grad(): 
            outputs = model(**inputs)
            print(text_id, var_id)