# import whisper

# model = whisper.load_model("base")  # "tiny", "small", "medium", "large" なども選べます
# result = model.transcribe("asano.wav")
# print(result["text"])
# with open("./answer_data/output_kakidashi.txt", "w", encoding="utf-8") as f:
#     f.write(result["text"])


import whisper
import os

def transcribe_audio(wav_path, output_path, model_name="base"):
    """
    音声ファイルを文字起こししてファイルに保存する関数

    Args:
        wav_path (str): 入力音声ファイルのパス
        output_path (str): 文字起こし結果を保存するテキストファイルのパス
        model_name (str): Whisperモデル名 ("tiny", "small", "medium", "large", "base" など)
    """
    # モデルロード
    model = whisper.load_model(model_name)

    # 推論
    result = model.transcribe(wav_path)
    text = result["text"]
    print("Transcription:", text)

    # ディレクトリがなければ作成

    # ファイルに書き込み
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    return text

# 使い方例
if __name__ == "__main__":
    transcribe_audio("asano.wav", "./answer_data/output_moziokoshi.txt")