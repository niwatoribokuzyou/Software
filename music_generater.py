from fastapi import FastAPI, BackgroundTasks, HTTPException 
from pydantic import BaseModel
import uuid
import base64
import time
from ollama_test import generate_bgm_prompt
from effb2_test import generate_audio_caption
from whisper_test import transcribe_audio
from scipy.io import wavfile

def generate_music(audio_data, room_temperature, room_illuminance):




	#本多ふぇーずはじまり

    caption_path = "./answer_data/output_caption.txt"
    moziokoshi_path = "./answer_data/output_moziokoshi.txt"
    

    # room_temperature = 24  # ℃（快適な冷房を効かせたリビング）
    # room_illuminance = 500  # lx（日中の自然光や照明で明るい状態）


    # 音源からキャプションを生成する関数
    # 第一引数音源バイナリデータ、第二引数キャプションを書き込むテキストファイルのパス
    caption = generate_audio_caption(audio_data, caption_path)

    # 音源を文字お越しする関数
    # 第一引数音源バイナリデータ、第二引数文字起こししたものを書き込むテキストファイルのパス
    moziokoshi = transcribe_audio(audio_data, moziokoshi_path)

    # キャプションと文字起こしからプロンプトを作るパス
    # 第一引数は文字起こしテキストファイルのパス、第二引数はキャプションテキストファイルのパス、第三引数はプロンプトを書き込むテキストファイルのパス
    # 第四引数は部屋の温度、第五引数は部屋の照度
    prompt = generate_bgm_prompt(moziokoshi_path,caption_path, room_temperature, room_illuminance)


    # music = suno(prompt)
    return prompt

if __name__ == "__main__":
    room_temperature = 24  # ℃（快適な冷房を効かせたリビング）
    room_illuminance = 500  # lx（日中の自然光や照明で明るい状態）
    with open("asano.wav", "rb") as f:
        audio_bytes = f.read()
    # sr, audio_data = wavfile.read("asano.wav")  # sr: サンプリングレート, audio_data: ndarray
    # print("audio_data",audio_data)
    # print("audio_data.shape",audio_data.shape)
    
    music = generate_music(audio_bytes, room_temperature, room_illuminance)
    print("music", music)