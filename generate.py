import torch
from transformers import pipeline
import torchaudio
import numpy as np
import time  # 追加
import io
import numpy as np
import soundfile as sf
from pydub import AudioSegment
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
    # if audio.ndim == 1:
    #     audio = audio[np.newaxis, :]
    if audio.ndim == 1 or audio.shape[0] == 1:
        audio = audio.flatten()
    else:
        audio = audio.T  # (samples, channels)
    
        # audio = audio[:, np.newaxis]
    # re_audio = audio.squeeze(0)

    # # float32 → int16
    # audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    # サンプルレートを手動で指定（MusicGenは通常32kHz）
    samplerate = 32000

    # torchaudio.save(
    #     "output-bed.wav",
    #     torch.from_numpy(audio_int16),
    #     samplerate
    # )
    # ---- WAVに一旦変換 ----
    buffer_wav = io.BytesIO()
    sf.write(buffer_wav, audio, samplerate, format="WAV")
    buffer_wav.seek(0)

    # ---- WAV → MP3に変換 ----
    segment = AudioSegment.from_file(buffer_wav, format="wav")
    buffer_mp3 = io.BytesIO()
    segment.export(buffer_mp3, format="mp3")
    buffer_mp3.seek(0)
    print("✅ output-bed.wav を保存しました！")
    # sunoの処理をここに書く
    # music = suno(prompt)
    return buffer_mp3.read(), samplerate

if __name__ == "__main__":
    prompt = "A dark, ambient electronic piece with slow tempo, featuring synthesizers and soft beeping sounds, reflecting the dim 10lx room. Genre: ambient electronic with a touch of jazz."

    music = generates(prompt)
    # print("music", music)