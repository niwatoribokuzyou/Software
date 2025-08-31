import torch
from transformers import pipeline
import torchaudio
import numpy as np
import time
import os

# -------------------------
# 設定
# -------------------------
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

pipe = pipeline("text-to-audio", model="facebook/musicgen-small", device=device)

# 出力フォルダ
out_dir = "generated_music"
os.makedirs(out_dir, exist_ok=True)

# 8ジャンルとそのプロンプト
genre_prompts = {
    "jazz": "Smooth jazz with saxophone, double bass, soft drums, and piano improvisation.",
    "rock": "Energetic rock with electric guitar riffs, bass, and powerful drums.",
    "classical": "Elegant classical symphony with violins, cellos, and a full orchestra.",
    "pop": "Catchy pop tune with bright synths, guitar, and a steady beat.",
    "hiphop": "Hip hop beat with deep bass, rhythmic drums, and subtle background samples.",
    "reggae": "Relaxed reggae with offbeat guitar, smooth bassline, and light percussion.",
    "blues": "Soulful blues with electric guitar, slow groove, and emotional feel.",
    "metal": "Heavy metal with distorted guitars, fast drums, and intense energy.",
}

# -------------------------
# 生成ループ
# -------------------------
for genre, prompt in genre_prompts.items():
    print(f"\n🎵 Generating {genre} ...")
    start_time = time.time()

    music = pipe(
        prompt,
        forward_params={"do_sample": True, "max_new_tokens": 1536}  # 約30秒前後
    )

    end_time = time.time()
    print(f"⏱ {genre} generation time: {end_time - start_time:.2f} sec")

    # (1, samples) → (channels, samples)
    audio = music["audio"].squeeze(0)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]

    # float32 → int16
    audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    # サンプルレート（MusicGenは通常32kHz）
    samplerate = 32000

    # 保存
    out_path = os.path.join(out_dir, f"{genre}.wav")
    torchaudio.save(
        out_path,
        torch.from_numpy(audio_int16),
        samplerate
    )

    print(f"✅ Saved: {out_path}")

print("\n🎶 All 8 genres generated successfully!")
