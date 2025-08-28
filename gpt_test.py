from openai import OpenAI
from dotenv import load_dotenv
import os

def chat_with_gpt(stt_data, caption, temperature, humidity, pressure, illuminance, model="gpt-4o-mini"):
    """
    ChatGPT API を使って応答を生成する関数
    """
    load_dotenv()

# 環境変数からAPIキーを取得
    api_key = os.getenv("OPEN_AI_APIKEY")

# クライアントを初期化（環境変数からAPIキーを自動で読み込み）
    client = OpenAI(api_key=api_key)
    prompt = f"""
以下の文字起こしとキャプションは同一音声から作り出されたものです。sunoでBGMを作成するためのテキストプロンプトを作ってください。
文字起こし内容:
「{stt_data}」

音声キャプション内容:
「{caption}」

BGMを流す部屋の環境
- 温度:{temperature}℃
- 湿度:{humidity}%
- 気圧:{pressure}hPa
- 照度:{illuminance}lx


上記から、音楽の雰囲気、テンポ、楽器、ジャンルなどを具体的に想像して、Sunoで使えるBGMプロンプトを提案してください。
生成する文字はプロンプトの文字だけでお願いします。 英語で140文字以下にしてください
"""
    

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    temperature = 24  # ℃（快適な冷房を効かせたリビング）
    humidity = 50  # %（快適な湿度）
    pressure = 1013  # hPa（標準気圧）
    illuminance = 500  # lx（日中の自然光や照明で明るい状態）

    # 文字起こしを書き込んだファイル
    stt_path="./answer_data/output_moziokoshi.txt" 
    with open(stt_path, 'r', encoding='utf-8') as f:
        stt_data = f.read()
    # キャプションを書き込んだファイル
    caption_path="./answer_data/output_caption.txt"
    with open(caption_path, 'r', encoding='utf-8') as f:
        caption = f.read()
    # BGM用プロンプトを書き込んだファイル
    output_path="./answer_data/output_bgm_prompt.txt"
    prompt = chat_with_gpt(
        stt_data,
        caption,
        temperature,
        humidity,
        pressure,
        illuminance,
    )
    print("Assistant:", prompt)