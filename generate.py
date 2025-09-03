import torch
from transformers import pipeline
import torchaudio
import numpy as np
import time  # 追加
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment


# BGM生成
def generates(prompt):
    
    device = 0
    print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)

    

    start_time = time.time()  # ここから計測
    music = pipe(
        prompt,
        forward_params={"do_sample": True, "max_new_tokens": 512}
    )
    end_time = time.time()  # ここまで計測

    print(f"⏱ 音楽生成にかかった時間: {end_time - start_time:.2f} 秒")

    # (1, samples) → (channels, samples)
    audio = music["audio"].squeeze(0)

    if audio.ndim == 1 or audio.shape[0] == 1:
        audio = audio.flatten()
    else:
        audio = audio.T  # (samples, channels)

    # サンプルレートを手動で指定（MusicGenは通常32kHz）
    samplerate = 32000


    # ---- WAVに一旦変換 ----
    buffer_wav = io.BytesIO()
    sf.write(buffer_wav, audio, samplerate, format="WAV")
    buffer_wav.seek(0)

    # ---- WAV → MP3に変換 ----
    segment = AudioSegment.from_file(buffer_wav, format="wav")
    buffer_mp3 = io.BytesIO()
    segment.export(buffer_mp3, format="mp3")
    buffer_mp3.seek(0)
    output_filename = f"output.mp3"
    with open(output_filename, "wb") as f:
        f.write(buffer_mp3.getbuffer())
    print("✅BGMを保存しました！")

    return buffer_mp3.read(), samplerate

if __name__ == "__main__":
    prompt = "A relaxed jazz piece with a smooth, mellow tempo, featuring soft piano, gentle saxophone, and light percussion. Suitable for a cozy night with a cheerful crowd, enhancing warmth and comfort."

    music = generates(prompt)
