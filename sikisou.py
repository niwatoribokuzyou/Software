# import matplotlib.pyplot as plt
# import numpy as np
# import colorsys

# # 色相を0〜360度で作る
# hues = np.linspace(0, 360, 360)

# # HSV -> RGB に変換
# colors = [colorsys.hsv_to_rgb(h/360, 1, 1) for h in hues]

# # 円グラフ用のデータ（すべて同じ値にして円周を埋める）
# values = np.ones_like(hues)

# fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))

# # 角度をラジアンに変換
# angles = np.deg2rad(hues)

# # 円グラフ描画
# ax.bar(angles, values, width=np.deg2rad(1), color=colors, edgecolor='none')

# ax.set_xticks([])
# ax.set_yticks([])
# ax.set_title("Color Hue Circle (0°〜360°)")
# plt.show()



import librosa
import colorsys
import time

def bpm_to_hue(bpm, bpm_min=60, bpm_max=180):
    # bpm_min → 寒色, bpm_max → 暖色
    bpm = max(min(bpm, bpm_max), bpm_min)
    # 240°(青) → 30°(オレンジ)
    hue = 240 - (bpm - bpm_min) / (bpm_max - bpm_min) * 210
    return hue / 360  # 0-1 に正規化

# 音楽ロード
y, sr = librosa.load("music.wav")
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
print("BPM:", tempo)

# HSV -> RGB に変換して LED 更新
hue = bpm_to_hue(tempo)
r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
print("RGB:", int(r*255), int(g*255), int(b*255))
