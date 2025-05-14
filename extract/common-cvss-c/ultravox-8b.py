import os
import numpy as np
import librosa
import torch
import pandas as pd
from transformers import AutoProcessor, AutoModel

folder_name = "ultravox-0_5-3_1-8b"
device = "cuda:1"

def hook_fn(m, i, o, layer_id):
    save_path = os.path.join(save_dir, f"{layer_id}.pt")
    torch.save(o[0][0, -1, :].detach().cpu(), save_path)
    
model = AutoModel.from_pretrained(
    "fixie-ai/ultravox-v0_5-llama-3_1-8b",
    cache_dir=f"../../../huggingface/{folder_name}",
    device_map={"":device},
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "fixie-ai/ultravox-v0_5-llama-3_1-8b",
    cache_dir=f"../../../huggingface/{folder_name}",
    trust_remote_code=True
)

for i, layer in enumerate(model.language_model.model.layers):
    layer.register_forward_hook(
        lambda m, i, o, layer_id=i: hook_fn(m, i, o, layer_id=layer_id)
    )

for language_code in ['de', 'es', 'fr', 'ja', 'zh-CN']:
    summary_df = pd.read_csv(f'../../../dataset-summary/common-cvss-c/{language_code}.csv')
    
    print('Starting to extract...')
    for index, row in summary_df.iloc[:50].iterrows():
        print(index)
        lang_code = 'zh' if language_code == 'zh-CN' else language_code
        
        mode = 'text'
        save_dir = f"../../../activations/cvss-c/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        sentence = f"{row['text']}"
        inputs = processor(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_outputs = model(**inputs)

        mode = 'raw'
        save_dir = f"../../../activations/cvss-c/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        prompt = f"<|audio|>"
        audio, sr = librosa.load(row['file'], sr=processor.audio_processor.feature_extractor.sampling_rate)
        inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_outputs = model(**inputs)
            
        mode = 'normalized'
        save_dir = f"../../../activations/cvss-c/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        prompt = f"<|audio|>"
        audio, sr = librosa.load(row['file'], sr=processor.audio_processor.feature_extractor.sampling_rate)
        audio = audio / np.abs(audio).max()
        inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_outputs = model(**inputs)
        
        mode = 'text'
        save_dir = f"../../../activations/common-voice/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        sentence = f"{row['sentence']}"
        inputs = processor(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_outputs = model(**inputs)

        mode = 'raw'
        save_dir = f"../../../activations/common-voice/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        prompt = f"<|audio|>"
        audio, sr = librosa.load(row['path'], sr=processor.audio_processor.feature_extractor.sampling_rate)
        inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt").to(device)
        outputs = model(**inputs)
        with torch.no_grad():
            audio_outputs = model(**inputs)
            
        mode = 'normalized'
        save_dir = f"../../../activations/common-voice/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        prompt = f"<|audio|>"
        audio, sr = librosa.load(row['path'], sr=processor.audio_processor.feature_extractor.sampling_rate)
        audio = audio / np.abs(audio).max()
        inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt").to(device)
        outputs = model(**inputs)
        with torch.no_grad():
            audio_outputs = model(**inputs)