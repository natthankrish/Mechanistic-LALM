import os
import numpy as np
import librosa
import torch
import pandas as pd
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

folder_name = "qwen2-audio-7b"
device = "cuda:2"

def hook_fn(m, i, o, layer_id):
    save_path = os.path.join(save_dir, f"{layer_id}.pt")
    torch.save(o[0][0, -1, :].detach().cpu(), save_path)

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B",
    cache_dir=f'../../../huggingface/{folder_name}/',
    trust_remote_code=True,
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

for language_code in ['de', 'es', 'fr', 'ja', 'zh-CN']:
    summary_df = pd.read_csv(f'../../../dataset-summary/common-cvss-t/{language_code}.csv')
    model = model.half()
    model.to(device)

    print('Starting to extract...')
    for index, row in summary_df.iloc[:50].iterrows():
        print(index)
        lang_code = 'zh' if language_code == 'zh-CN' else language_code
        
        mode = 'text'
        save_dir = f"../../../activations/cvss-t/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        sentence = f"{row['text']}"
        inputs = processor(sentence, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_outputs = model(**inputs)

        mode = 'raw'
        save_dir = f"../../../activations/cvss-t/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>"
        audio, sr = librosa.load(row['file'], sr=processor.feature_extractor.sampling_rate)
        inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt").to(device)
        with torch.no_grad():
            audio_outputs = model(**inputs)
            
        mode = 'normalized'
        save_dir = f"../../../activations/cvss-t/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>"
        audio, sr = librosa.load(row['file'], sr=processor.feature_extractor.sampling_rate)
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
        prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>"
        audio, sr = librosa.load(row['path'], sr=processor.feature_extractor.sampling_rate)
        inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt").to(device)
        outputs = model(**inputs)
        with torch.no_grad():
            audio_outputs = model(**inputs)
            
        mode = 'normalized'
        save_dir = f"../../../activations/common-voice/{folder_name}/{lang_code}/{row['id']}/{mode}"
        os.makedirs(save_dir, exist_ok=True)
        prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>"
        audio, sr = librosa.load(row['path'], sr=processor.feature_extractor.sampling_rate)
        audio = audio / np.abs(audio).max()
        inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt").to(device)
        outputs = model(**inputs)
        with torch.no_grad():
            audio_outputs = model(**inputs)