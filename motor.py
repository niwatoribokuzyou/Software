
import librosa
import pygame
import time
import numpy as np
# MP3読み込み（ビート検出用）
music_path = "./output-f.mp3"
y, sr = librosa.load(music_path, sr=None)

# ビート検出
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

# tempo が配列になっている場合に対応
if isinstance(tempo, (list, tuple, np.ndarray)):
    tempo_value = float(tempo[0])
else:
    tempo_value = float(tempo)

beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print(f"検出されたビート数: {len(beat_times)}, BPM: {tempo_value:.2f}")

# pygameで音楽再生
pygame.mixer.init(frequency=sr)
pygame.mixer.music.load(music_path)
pygame.mixer.music.play()

# クールタイム設定
cool_time = 4  # 秒
last_action_time = -float('inf')
start_time = time.time()

# ビートに合わせてモーター動作（print）
for i, beat_time in enumerate(beat_times):
    sleep_time = beat_time - (time.time() - start_time)
    if sleep_time > 0:
        time.sleep(sleep_time)

    if time.time() - last_action_time >= cool_time:
        print(f"ビート {i+1} → モーターが動く！")
        last_action_time = time.time()
    else:
        print(f"ビート {i+1} → クールタイム中でスキップ")

# 音楽再生終了まで待機
while pygame.mixer.music.get_busy():
    time.sleep(0.1)

print("音楽再生終了")
