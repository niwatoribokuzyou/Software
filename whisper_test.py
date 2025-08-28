
import whisper
import os

def transcribe_audio(audio_bytes, wav_path, output_path, model_name="base"):
    """
    音声ファイルを文字起こししてファイルに保存する関数

    Args:
        wav_path (str): 入力音声ファイルのパス
        output_path (str): 文字起こし結果を保存するテキストファイルのパス
        model_name (str): Whisperモデル名 ("tiny", "small", "medium", "large", "base" など)
    """
    # モデルロード
    with open(wav_path, "wb") as f:
        f.write(audio_bytes)
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
    wav_path = "asuno.wav"
    with open("asano.wav", "rb") as f:
        audio_bytes = f.read()
    transcribe_audio(audio_bytes,wav_path, "./answer_data/output_moziokoshi.txt")

# import whisper
# import numpy as np
# import io
# import soundfile as sf 
# import librosa

# def transcribe_audio_bytes(audio_bytes, model_name="base"):
#     """
#     バイナリ音声データから文字起こし
#     """
#     # バイトデータを NumPy 配列に読み込む
#     audio_file = io.BytesIO(audio_bytes)
#     audio_data, sr = sf.read(audio_file)  # audio_data: np.ndarray, sr: サンプリングレート

#     # Whisper モデルロード
#     model = whisper.load_model(model_name)

#     # Whisper は 16kHz で推論することが推奨なので、必要に応じてリサンプリング
#     if sr != 16000:
#         audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)

#     # 推論
#     result = model.transcribe(audio_data)
#     text = result["text"]
#     print("Transcription:", text)

#     return text

# if __name__ == "__main__":
#     # 例: ラズパイから受け取った音声バイナリ
#     with open("asano.wav", "rb") as f:
#         audio_bytes = f.read()
#     transcribe_audio_bytes(audio_bytes, "./answer_data/output_moziokoshi.txt")