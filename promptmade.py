# 音源からキャプションと文字起こしをしてollamaでBGMのプロンプトを作成するファイル
from gpt_test import chat_with_gpt
# from ollama_test import generate_bgm_prompt
from effb2_test import generate_audio_caption
from whisper_test import transcribe_audio
# voice_path = f"./snippets/voice_chunk_{i+1}.wav"

if __name__ == "__main__":
    for i in range(4):
        voice_path = f"./{i+1}.wav"

        print("START")
        print("voice_path",voice_path)
        with open(voice_path, "rb") as f:
            audio_bytes = f.read()
        # caption_path = "./answer_data/output_caption.txt"
        # moziokoshi_path = "./answer_data/output_moziokoshi.txt"
        # bgm_prompt_path = "./answer_data/output_bgm_prompt.txt"
        # # 部屋の環境情報（例：ベッドルーム）
        # room_temperature = 26  # ℃ (夏のベッドルームの温度)
        # room_illuminance = 30  # lx（就寝前リラックス用の低照度）

        # 部屋の環境情報（例：リビング・昼）
        # room_temperature = 24  # ℃（快適な冷房を効かせたリビング）
        # room_illuminance = 500  # lx（日中の自然光や照明で明るい状態）


        # temperature = 27#夜ベッド
        # humidity = 60
        # pressure = 1012
        # illuminance = 10

        temperature = 28#昼リビング
        humidity = 55
        pressure = 1012
        illuminance = 500

        # 音源からキャプションを生成する関数
        # 第一引数音源バイナリデータ、第二引数キャプションを書き込むテキストファイルのパス
        caption = generate_audio_caption(audio_bytes)

        # 音源を文字お越しする関数
        # 第一引数音源バイナリデータ、第二引数文字起こししたものを書き込むテキストファイルのパス
        stt_data = transcribe_audio(audio_bytes)

        # キャプションと文字起こしからプロンプトを作るパス
        # 第一引数は文字起こしテキストファイルのパス、第二引数はキャプションテキストファイルのパス、第三引数はプロンプトを書き込むテキストファイルのパス
        # 第四引数は部屋の温度、第五引数は部屋の照度
        prompt = chat_with_gpt(stt_data, caption, temperature, humidity, pressure, illuminance, model="gpt-4o")
        print("prompt:ああああああああああああああああ", prompt)
