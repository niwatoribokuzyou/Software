import torch
from transformers import pipeline
import torchaudio
import numpy as np
import time  # 追加
# from test_generate_combine import combine_musics
import soundfile as sf
import io
import base64
from generate import generates
from lets_music_with_beat import blend_soundscape_music
def generate_music(prompt, kankyouonn):
    
    
    # sunoの処理をここに書く
    # music = suno(prompt)

    #たぶん
    music, sr = generates(prompt)

    combine_music = blend_soundscape_music(music, kankyouonn)
    # buffer = io.BytesIO()
#     sf.write(buffer, music, sr, format='WAV')
#     buffer.seek(0)
# # バイト列を base64 に変換
    encoded_audio = base64.b64encode(combine_music)
    return encoded_audio

if __name__ == "__main__":
    prompt = "Generated BGM Prompt"

    music = generate_music(prompt)
    print("music", music)