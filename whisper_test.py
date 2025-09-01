from pydub import AudioSegment
import io
import whisper

def transcribe_audio(audio_bytes, model_name="base"):
    """
    音声ファイルを文字起こししてファイルに保存する関数

    Args:
        audio_bytes (bytes): 音声データのバイナリ
        model_name (str): Whisperモデル名 ("tiny", "small", "medium", "large", "base" など)
    """
    # print("aaaaaaaaaaaaaaaaaaa")
    input_path = "input_stt.wav"
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    # print("bbbbbbbbbbbbbbbbb")

    audio.export(input_path, format="wav")
    # print("cccccccccccccccccccccc")
    
    # モデルロード
    model = whisper.load_model(model_name)
    # print("ddddddddddddddddd")

    # 推論
    result = model.transcribe(input_path)
    # print("eeeeeeeeeeeeeeeeeeeeee")

    text = result["text"]
    print("Transcription:", text)

    return text

# 使い方例
if __name__ == "__main__":
    with open("input_caption.mp3", "rb") as f:
        audio_bytes = f.read()
    transcribe_audio(audio_bytes)

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