import torch
from transformers import pipeline
import torchaudio
import numpy as np
import time  # 追加

def generates(prompt):
    
    device = 0
    print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)

    

    start_time = time.time()  # ここから計測
    music = pipe(
        prompt,
        forward_params={"do_sample": True, "max_new_tokens": 1536}
    )
    end_time = time.time()  # ここまで計測

    print(f"⏱ 音楽生成にかかった時間: {end_time - start_time:.2f} 秒")

    # (1, samples) → (channels, samples)
    audio = music["audio"].squeeze(0)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    re_audio = audio.squeeze(0)

    # float32 → int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    # サンプルレートを手動で指定（MusicGenは通常32kHz）
    samplerate = 32000

    torchaudio.save(
        "output-bed.wav",
        torch.from_numpy(audio_int16),
        samplerate
    )

    print("✅ output-bed.wav を保存しました！")
    # sunoの処理をここに書く
    # music = suno(prompt)
    return re_audio, samplerate

# if __name__ == "__main__":
    # prompt = "Generated BGM Prompt"

    # music = generate_music(prompt)
    # print("music", music)