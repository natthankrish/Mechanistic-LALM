import torch
import os
from transformers import AutoModel, AutoProcessor

folder_name = "ultravox-0_5-3_2-1b"

def hook_fn(m, i, o, layer_id):
    save_dir = f"../../../activations/mvicuna/{folder_name}/{current_language}/{text_id}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{layer_id}.pt")
    torch.save(o[0][0, -1, :].detach().cpu(), save_path)
    
model = AutoModel.from_pretrained(
    "fixie-ai/ultravox-v0_5-llama-3_2-1b",
    cache_dir=f"../../../huggingface/{folder_name}",
    device_map={"":"cuda:0"},
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "fixie-ai/ultravox-v0_5-llama-3_2-1b",
    cache_dir=f"../../../huggingface/{folder_name}",
    trust_remote_code=True
)

for i, layer in enumerate(model.language_model.model.layers):
    layer.register_forward_hook(
        lambda m, i, o, layer_id=i: hook_fn(m, i, o, layer_id=layer_id)
    )

languages = ['en', 'es', 'fr', 'id', 'ja', 'vi', 'zh']
for current_language in languages:
    text_id = 0
    with open(f"../../../input/mvicuna/{current_language}.txt", "r") as file:
        for line in file:
            prompt = line.strip()

            inputs = processor(text=prompt, audios=[], return_tensors="pt")
            inputs.to("cuda:0")
            
            with torch.no_grad(): 
                outputs = model(**inputs)
                print(text_id)
         
            text_id = text_id + 1