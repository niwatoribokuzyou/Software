import io
from fastapi import FastAPI, BackgroundTasks, HTTPException 
from pydantic import BaseModel
import uvicorn
import uuid
import base64
from gpt_test import chat_with_gpt
from effb2_test import generate_audio_caption
from whisper_test import transcribe_audio
from music_generater import generate_music
# WAVファイルをバイト列として読み込み
with open("./voice.wav", "rb") as f:
    decoded_audio = f.read()

# これで base64.b64decode(audio_data) と同じ状態になる
# y が音声データ（numpy配列）なので decoded_audio の代わりに使える
caption = generate_audio_caption(decoded_audio)
stt_data = transcribe_audio(decoded_audio)
temperature = 27#夜ベッド
humidity = 60
pressure = 1012
illuminance = 10
# temperature = env_data.get("temperature", 25)
# pressure = env_data.get("pressure", 1013)
# humidity = env_data.get("humidity", 50)
# illuminance = env_data.get("lux", 500)

prompt = chat_with_gpt(stt_data, caption, temperature, humidity, pressure, illuminance)
print("Generated Prompt:", prompt)

# Sunoで音楽生成
music = generate_music(prompt, decoded_audio)

print("musicFINISH")
