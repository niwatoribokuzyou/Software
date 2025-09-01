from audio_slice_kansuu import process_audio_files
import numpy as np
import soundfile as sf
from lets_music_with_beat import overlay_short_sounds_numpy
kankyouonn = process_audio_files("./voice.wav")
print("kankyouonn", len(kankyouonn))
output_path  = "./output/kankyou.wav"
sr = 16000

overlay_short_sounds_numpy(BGM, sr, kankyouonn[0], output_path)

for i in range(len(kankyouonn)):
    sf.write(f"./output/kankyou{i}.wav", kankyouonn[i], sr)
    print(f"✅ 保存しました: {output_path}")