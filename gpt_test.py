from openai import OpenAI
from dotenv import load_dotenv
import os
import json

from datetime import datetime
from wheather import get_weather_by_location
def chat_with_gpt(stt_data, caption, temperature, humidity, pressure, illuminance, model="gpt-4o"):
    """
    ChatGPT API を使って応答を生成する関数
    """
    load_dotenv()
    now = datetime.now()
    hour = now.hour
    if 5 <= hour < 11:
        tod = "朝"
    elif 11 <= hour < 17:
        tod = "昼"
    elif 17 <= hour < 20:
        tod = "夕方"
    else:
        tod = "夜"
    datetime_info = now.strftime(f"%Y-%m-%d %H:%M {tod}")
    # print("datetime_info",datetime_info)

    wheather_imformation = get_weather_by_location()
# 環境変数からAPIキーを取得
    api_key = os.getenv("OPEN_AI_APIKEY")

# クライアントを初期化（環境変数からAPIキーを自動で読み込み）
    client = OpenAI(api_key=api_key)

    # prompt = f"""
    # 以下の文字起こしとキャプションは同一音声から作られたものです。text-to-audioの"facebook/musicgen-small"でBGMを作成するためのテキストプロンプトを作ってください。
    # 文字起こし内容:
    # 「{stt_data}」

    # 音声キャプション内容:
    # 「{caption}」

    # BGMを流す部屋の環境:
    # - 日時:{datetime_info}
    # - 天気:{wheather_imformation}
    # - 温度:{temperature}℃
    # - 湿度:{humidity}%
    # - 気圧:{pressure}hPa
    # - 照度:{illuminance}lx

    # 上記から、音楽の雰囲気、テンポ、楽器、ジャンルを具体的に想像して、text-to-audioの"facebook/musicgen-small"というモデルで使えるBGMプロンプトを提案してください。
    # 照度は特にBGMを流す空間のムードを反映するので重視してください。
    # 明るい場所では明るいBGM、暗い場所では暗いBGMにするようにしてください。
    # 最も自然なジャンルを jazz, rock, classical, pop, hiphop, reggae, blues, metal から選んで入れてください。
    # 出力はプロンプト文のみ、英語で150文字以内にしてください。
    # """


    prompt = f"""
    以下の文字起こしとキャプションは同一音声から作られたものです。text-to-audioの"facebook/musicgen-small"でBGMを作成するためのテキストプロンプトを作ってください。
    文字起こし内容:
    「{stt_data}」

    音声キャプション内容:
    「{caption}」

    BGMを流す部屋の環境:
    - 日時:{datetime_info}
    - 天気:{wheather_imformation}
    - 温度:{temperature}℃
    - 湿度:{humidity}%
    - 気圧:{pressure}hPa
    - 照度:{illuminance}lx

    上記から、音楽の雰囲気、テンポ、楽器、ジャンルを具体的に想像して、text-to-audioの"facebook/musicgen-small"で使えるBGMプロンプトを提案してください。
    さらに、この曲にあったLEDのイメージカラー範囲も決めてください。min_colorとmax_colorでJSON形式で出力してください。
    出力例:
    {{"bgm_prompt":"...", "min_color":"#RRGGBB", "max_color":"#RRGGBB"}}
    bgm_promptは英語で150文字以内にしてください。
    """
    

    print("prompt",prompt)
    # exit()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=200
    )

    prompt = response.choices[0].message.content
    data = json.loads(prompt)
    bgm_prompt = data["bgm_prompt"]
    print("bgm_prompt",bgm_prompt)
    # 色だけ取り出す

    # print("data[min_color]", data["min_color"])
    # print("data[max_color]", data["max_color"])

    min_color_str = str(data["min_color"])
    max_color_str = str(data["max_color"])

    print("min_color11111:", min_color_str)
    # print("min_color:", min_color_str)

    print("max_color11111:", max_color_str)

    return bgm_prompt, min_color_str, max_color_str


if __name__ == "__main__":
    temperature = 27#夜ベッド
    humidity = 60
    pressure = 1012
    illuminance = 10

    # temperature = 28#昼リビング
    # humidity = 55
    # pressure = 1012
    # illuminance = 500

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
    prompt, min_color, max_color = chat_with_gpt(
        stt_data,
        caption,
        temperature,
        humidity,
        pressure,
        illuminance,
    )
    print("prompt",prompt)
    print("min_color",min_color)
    print("max_color",max_color)

    




# from openai import OpenAI
# from dotenv import load_dotenv
# import os
# from generate import generates
# def chat_with_gpt(stt_data, caption, temperature, humidity, pressure, illuminance, datetime_info, weather_info, model="gpt-4o"):
#     """
#     ChatGPT API を使ってBGMプロンプトを生成する関数
#     datetime_info: 任意の日時文字列 (例: "2025-08-01 13:00 昼")
#     weather_info: 任意の天気文字列 (例: "晴れ")
#     """
#     load_dotenv()
#     api_key = os.getenv("OPEN_AI_APIKEY")

#     client = OpenAI(api_key=api_key)

#     prompt = f"""
#     以下の文字起こしとキャプションは同一音声から作られたものです。text-to-audioの"facebook/musicgen-small"でBGMを作成するためのテキストプロンプトを作ってください。
#     文字起こし内容:
#     「{stt_data}」

#     音声キャプション内容:
#     「{caption}」

#     BGMを流す部屋の環境:
#     - 日時:{datetime_info}
#     - 天気:{weather_info}
#     - 温度:{temperature}℃
#     - 湿度:{humidity}%
#     - 気圧:{pressure}hPa
#     - 照度:{illuminance}lx

#     上記から、音楽の雰囲気、テンポ、楽器、ジャンルを具体的に想像して、text-to-audioの"facebook/musicgen-small"というモデルで使えるBGMプロンプトを提案してください。
#     照度は特にBGMを流す空間のムードを反映するようにしてください。
#     明るい場所では明るいBGM、暗い場所では暗いBGMにするようにしてください。
#     最も自然なジャンルを jazz, rock, classical, pop, hiphop, reggae, blues, metal から選んで入れてください。
#     出力はプロンプト文のみ、英語で150文字以内にしてください。
#     """
#     print("prompt::::::",prompt)
#     response = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.7,
#         max_tokens=200
#     )
#     return response.choices[0].message.content


# if __name__ == "__main__":
#     # 文字起こしファイル
#     stt_path = "./answer_data/output_moziokoshi.txt"
#     caption_path = "./answer_data/output_caption.txt"

#     with open(stt_path, 'r', encoding='utf-8') as f:
#         stt_data = f.read()
#     with open(caption_path, 'r', encoding='utf-8') as f:
#         caption = f.read()

#     pressure = 1012

#     scenarios = [
#         # 夏
#         {"season": "夏", "datetime_info": "2025-08-01 07:00 朝", "temperature": 26.5, "humidity": 50, "illuminance": 300, "weather": "晴れ"},
#         {"season": "夏", "datetime_info": "2025-08-01 13:00 昼", "temperature": 26.5, "humidity": 50, "illuminance": 500, "weather": "晴れ"},
#         {"season": "夏", "datetime_info": "2025-08-01 22:00 夜", "temperature": 26.5, "humidity": 50, "illuminance": 10, "weather": "晴れ"},
#         # 冬
#         {"season": "冬", "datetime_info": "2025-12-01 07:00 朝", "temperature": 22, "humidity": 35, "illuminance": 200, "weather": "晴れ"},
#         {"season": "冬", "datetime_info": "2025-12-01 13:00 昼", "temperature": 22, "humidity": 35, "illuminance": 500, "weather": "晴れ"},
#         {"season": "冬", "datetime_info": "2025-12-01 22:00 夜", "temperature": 22, "humidity": 35, "illuminance": 10, "weather": "晴れ"},
#     ]

#     for i, sc in enumerate(scenarios):
#         print(f"--- Scenario: {sc['season']} {sc['datetime_info']} illuminance={sc['illuminance']} ---")
#         prompt_text = chat_with_gpt(
#             stt_data,
#             caption,
#             sc["temperature"],
#             sc["humidity"],
#             pressure,
#             sc["illuminance"],
#             sc["datetime_info"],
#             sc["weather"]
#         )
#         generates(prompt_text, i)
#         print(prompt_text, i)
#         print("\n")
