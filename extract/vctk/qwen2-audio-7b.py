import torch
import os
import pandas as pd
import librosa
import numpy as np
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

folder_name = "qwen2-audio-7b"
device = "cuda:2"

def hook_fn(m, i, o, layer_id):
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
    
model = model.half()
model.to(device)

for mic in ['mic1', 'mic2']:
    df = pd.read_csv(f'../../../dataset-summary/vctk/{mic}.csv', low_memory=False)
    columns = [col for col in df.columns if col not in ['text_id', 'text', 'count_nan']]
    for index, row in df.iloc[:10].iterrows():
        text_id = row['text_id']
        for speaker_id in columns:
            file_path = df[(df['text_id'] == text_id)][speaker_id].values[0]
            transcript = df[(df['text_id'] == text_id)]['text'].values[0]
            print(text_id, speaker_id, file_path)

            if pd.notnull(file_path):
                mode = f"{mic}_raw"
                save_dir = f"../../../activations/vctk/{folder_name}/{text_id}/{mode}/{speaker_id}/"
                os.makedirs(save_dir, exist_ok=True)
                prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>"
                audio, sr = librosa.load(file_path, sr=processor.feature_extractor.sampling_rate)
                inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt")
                inputs.to(device)
                
                with torch.no_grad():
                    audio_outputs = model(**inputs)
                    
                mode = f"{mic}_normalized"
                save_dir = f"../../../activations/vctk/{folder_name}/{text_id}/{mode}/{speaker_id}/"
                os.makedirs(save_dir, exist_ok=True)
                audio = audio / np.abs(audio).max()
                inputs = processor(text=prompt, audios=[audio], sampling_rate=sr, return_tensors="pt")
                inputs.to(device)
                
                with torch.no_grad():
                    audio_outputs = model(**inputs)
                    
                mode = f"text"
                save_dir = f"../../../activations/vctk/{folder_name}/{text_id}/{mode}/"
                os.makedirs(save_dir, exist_ok=True)
                inputs = processor(text=transcript, return_tensors="pt")
                inputs.to(device)
                
                with torch.no_grad():
                    audio_outputs = model(**inputs)
