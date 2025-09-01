# send_audio.py
import requests
import base64
import json

# サーバーのURL
SERVER_URL = "http://localhost:8000/api/v1/data"

# 送信するMP3ファイル
MP3_FILE_PATH = "voice_chunk_2.mp3"

# 環境データの例
env_data = {
    "temperature": 26,
    "pressure": 1012,
    "humidity": 60,
    "lux": 300
}

def send_mp3(mp3_path: str, env_data: dict):
    # MP3をバイナリで読み込み
    with open(mp3_path, "rb") as f:
        audio_bytes = f.read()

    # base64に変換
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    payload = {
        "audio_data": audio_b64,
        "environmental_data": env_data
    }

    # POSTリクエスト
    response = requests.post(SERVER_URL, json=payload)

    if response.status_code == 202:
        print("送信成功！タスクID:", response.json()["task_id"])
    else:
        print("送信失敗:", response.status_code, response.text)

if __name__ == "__main__":
    send_mp3(MP3_FILE_PATH, env_data)
