# import sys
# import ollama

# # コマンドライン引数から質問を取得
# input_kakidashi = "./answer_data/output_kakidashi.txt"
# input_caption = "./answer_data/output_caption.txt"

# # ファイルから内容を読み込む
# with open(input_kakidashi, "r", encoding="utf-8") as f:
#     query_kakidashi = f.read().strip()
# with open(input_caption, "r", encoding="utf-8") as f:
#     query_caption = f.read().strip()

# query = f"""
# 以下の文字起こしとキャプションは同一音声から作り出されたものです。sunoでBGMを作成するためのテキストプロンプトを作ってください。
# 文字起こし内容:
# 「{query_kakidashi}」

# 音声キャプション内容:
# 「{query_caption}」

# 上記から、音楽の雰囲気、テンポ、楽器、ジャンルなどを具体的に想像して、Sunoで使えるBGMプロンプトを提案してください。
# 生成する文字はプロンプトの文字だけでおねがいします。
# """
# print("question:MOZI",query_kakidashi, "CAPTION", query_caption)

# # Ollama に送信
# response = ollama.chat(
#     model="gemma3:4b",
#     messages=[{"role": "user", "content": query}]
# )

# # 結果を表示
# print(response['message']['content'])
# with open("./answer_data/output_bgm_prompt.txt", "w", encoding="utf-8") as f:
#     f.write(query)



import os
import ollama

# def generate_bgm_prompt(kakidashi_path, caption_path, output_path, model_name="gemma3:4b"):
def generate_bgm_prompt(kakidashi_path, caption_path,   output_path, room_temperature, room_illuminance, model_name="gpt-oss:20b"):

    """
    文字起こしと音声キャプションを使って、SunoでBGMを作成するためのプロンプトを生成し、ファイルに保存する関数

    Args:
        kakidashi_path (str): 文字起こし結果のファイルパス
        caption_path (str): 音声キャプション結果のファイルパス
        output_path (str): プロンプトを保存するテキストファイルのパス
        model_name (str): Ollamaのモデル名
    """
    # ファイルから内容を読み込む
    with open(kakidashi_path, "r", encoding="utf-8") as f:
        query_kakidashi = f.read().strip()
    with open(caption_path, "r", encoding="utf-8") as f:
        query_caption = f.read().strip()

    
    # プロンプト作成
    query = f"""
以下の文字起こしとキャプションは同一音声から作り出されたものです。sunoでBGMを作成するためのテキストプロンプトを作ってください。
文字起こし内容:
「{query_kakidashi}」

音声キャプション内容:
「{query_caption}」

BGMを流す部屋の環境
- 温度:{room_temperature}℃
- 照度:{room_illuminance}lx


上記から、音楽の雰囲気、テンポ、楽器、ジャンルなどを具体的に想像して、Sunoで使えるBGMプロンプトを提案してください。
生成する文字はプロンプトの文字だけでお願いします。 英語で140文字以下にしてください
"""

    # print("question:MOZI", query_kakidashi, "CAPTION", query_caption)
    print("QQUERY", query)

    # Ollama に送信
    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": query}]
    )

    prompt_result = response['message']['content']
    print("Generated BGM Prompt:", prompt_result)

    # ディレクトリがなければ作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # プロンプトをファイルに書き込み
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(prompt_result)

    return prompt_result


# 使い方例
if __name__ == "__main__":
    # # 部屋の環境情報（例：ベッドルーム）
    # room_temperature = 26  # ℃ (夏のベッドルームの温度)
    # room_illuminance = 30  # lx（就寝前リラックス用の低照度）

    # 部屋の環境情報（例：リビング・昼）
    room_temperature = 24  # ℃（快適な冷房を効かせたリビング）
    room_illuminance = 500  # lx（日中の自然光や照明で明るい状態）

    # 文字起こしを書き込んだファイル
    kakidashi_path="./answer_data/output_moziokoshi.txt" 
    # キャプションを書き込んだファイル
    caption_path="./answer_data/output_caption.txt"
    # BGM用プロンプトを書き込んだファイル
    output_path="./answer_data/output_bgm_prompt.txt"


    generate_bgm_prompt(
        kakidashi_path,
        caption_path,
        output_path,
        room_temperature,
        room_illuminance,
    )
