from pydub import AudioSegment
import math
import os

def split_wav_by_seconds(input_file: str, output_dir: str, chunk_seconds: int = 60):
    """
    WAVファイルを指定した秒数ごとに分割して保存する関数。

    Args:
        input_file (str): 入力WAVファイルのパス
        output_dir (str): 出力ディレクトリ
        chunk_seconds (int): 分割する時間（秒単位、デフォルト60秒）
    """
    os.makedirs(output_dir, exist_ok=True)

    # 音声読み込み
    audio = AudioSegment.from_wav(input_file)

    # チャンクの長さ（ミリ秒）
    chunk_length_ms = chunk_seconds * 1000

    # 分割数
    num_chunks = math.ceil(len(audio) / chunk_length_ms)

    for i in range(num_chunks):
        start_ms = i * chunk_length_ms
        end_ms = min((i + 1) * chunk_length_ms, len(audio))
        chunk = audio[start_ms:end_ms]
        print("chunk:",chunk, "i:", i)
        output_file = os.path.join(output_dir, f"voice_chunk_{i+1}.wav")
        chunk.export(output_file, format="wav")
        print(f"Saved: {output_file}")
    
    return num_chunks, 

# 使い方例
if __name__ == "__main__":
    split_wav_by_seconds("voice.wav", "split_wavs", chunk_seconds=60)  # 30秒ごとに分割
