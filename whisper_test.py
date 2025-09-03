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
    input_path = "input_stt.wav"
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

    audio.export(input_path, format="wav")
    
    # モデルロード
    model = whisper.load_model(model_name)

    # 推論
    result = model.transcribe(input_path)

    text = result["text"]
    print("Transcription:", text)

    return text

# 使い方例
if __name__ == "__main__":
    with open("./voice_chunk_2.mp3", "rb") as f:
        audio_bytes = f.read()
    transcribe_audio(audio_bytes)
