# 音源からキャプションと文字起こしをしてollamaでBGMのプロンプトを作成するファイル

from ollama_test import generate_bgm_prompt
from effb2_test import generate_audio_caption
from whisper_test import transcribe_audio

if __name__ == "__main__":

    voice_path = "asano.wav"

    caption_path = "./answer_data/output_caption.txt"
    moziokoshi_path = "./answer_data/output_moziokoshi.txt"
    bgm_prompt_path = "./answer_data/output_bgm_prompt.txt"
    # 部屋の環境情報（例：ベッドルーム）
    room_temperature = 26  # ℃ (夏のベッドルームの温度)
    room_illuminance = 30  # lx（就寝前リラックス用の低照度）

    # 音源からキャプションを生成する関数
    # 第一引数音源のパス、第二引数キャプションを書き込むテキストファイルのパス
    generate_audio_caption(voice_path, caption_path)

    # 音源を文字お越しする関数
    # 第一引数音源のパス、第二引数文字起こししたものを書き込むテキストファイルのパス
    transcribe_audio(voice_path, moziokoshi_path)

    # キャプションと文字起こしからプロンプトを作るパス
    # 第一引数は文字起こしテキストファイルのパス、第二引数はキャプションテキストファイルのパス、第三引数はプロンプトを書き込むテキストファイルのパス
    # 第四引数は部屋の温度、第五引数は部屋の照度
    generate_bgm_prompt(moziokoshi_path,caption_path,bgm_prompt_path, room_temperature, room_illuminance)
