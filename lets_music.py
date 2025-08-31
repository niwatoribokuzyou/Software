


# from pydub import AudioSegment
# import numpy as np

# # 音源読み込み
# bgm = AudioSegment.from_file("./output-natsu-bed.wav")
# short_sound = AudioSegment.from_file("./鍵を開ける1.mp3")

# # BGMをnumpy配列に変換（分析用）
# bgm_samples = np.array(bgm.get_array_of_samples(), dtype=np.float32)
# bgm_samples /= np.max(np.abs(bgm_samples))  # 正規化

# window_ms = 50  # 分析の分解能
# window_size = int(bgm.frame_rate * window_ms / 1000)

# timings = []

# for i in range(0, len(bgm_samples), window_size):
#     window = bgm_samples[i:i+window_size]
#     if len(window) == 0:
#         continue
#     peak = np.max(np.abs(window))
#     if peak > 0.4:  # ある程度音がある時だけ短音を鳴らす
#         ms_position = int(i / bgm.frame_rate * 1000)
#         timings.append((ms_position, peak))  # ピークも記録

# # BGMに短音を重ねる
# output = bgm
# for t, peak in timings:
#     # BGMのピークに応じて短音の音量を変える
#     gain_db = 20 * np.log10(peak + 1e-6) + 6  # +6dBで少し大きめに
#     short_adjusted = short_sound + gain_db
#     output = output.overlay(short_adjusted, position=t)

# # 保存
# output.export("mixed_auto_adjusted.wav", format="wav")
# print("自動生成完了。mixed_auto_adjusted.wav に保存されました。")






# from pydub import AudioSegment
# import numpy as np

# # -----------------------------
# # 音源読み込み
# # -----------------------------
# bgm = AudioSegment.from_file("./output-natsu-bed.wav")       # 長めのBGM
# short_sound = AudioSegment.from_file("./鍵を開ける1.mp3")   # 短い音

# # -----------------------------
# # BGMをnumpy配列に変換（分析用）
# # -----------------------------
# bgm_samples = np.array(bgm.get_array_of_samples(), dtype=np.float32)
# bgm_samples /= np.max(np.abs(bgm_samples))  # 正規化

# window_ms = 50  # 分析の分解能（50ms）
# window_size = int(bgm.frame_rate * window_ms / 1000)

# # -----------------------------
# # 短音を鳴らすタイミングリスト作成
# # -----------------------------
# timings = []
# cooldown_ms = 150  # 一度短音を鳴らしたら150msはスキップ
# last_hit = -cooldown_ms
# prev_peak = 0.0

# for i in range(0, len(bgm_samples), window_size):
#     window = bgm_samples[i:i+window_size]
#     if len(window) == 0:
#         continue

#     peak = np.max(np.abs(window))
#     ms_position = int(i / bgm.frame_rate * 1000)

#     # 閾値を超え、かつ前のピークがある程度以上のときのみ短音を鳴らす
#     if prev_peak > 0.1 and peak - prev_peak > 0.3 and ms_position - last_hit >= cooldown_ms:
#         timings.append((ms_position, peak))
#         last_hit = ms_position

#     prev_peak = peak

# # -----------------------------
# # BGMに短音を重ねる
# # -----------------------------
# output = bgm
# for t, peak in timings:
#     # BGMのピークに応じて短音の音量を変える
#     gain_db = 20 * np.log10(peak + 1e-6) + 6  # +6dBで少し大きめに
#     short_adjusted = short_sound + gain_db
#     output = output.overlay(short_adjusted, position=t)

# # -----------------------------
# # 保存
# # -----------------------------
# output.export("mixed_auto_filtered.wav", format="wav")
# print("自動生成完了。mixed_auto_filtered.wav に保存されました。")

from pydub import AudioSegment
import numpy as np

# -----------------------------
# 音源読み込み
# -----------------------------
bgm = AudioSegment.from_file("./output-kankyousubete.wav")       # 長めのBGM
short_sound = AudioSegment.from_file("./鍵を開ける1.mp3")   # 短い音

# -----------------------------
# BGMをnumpy配列に変換（分析用）
# -----------------------------
bgm_samples = np.array(bgm.get_array_of_samples(), dtype=np.float32)
bgm_samples /= np.max(np.abs(bgm_samples))  # 正規化

window_ms = 50  # 分析の分解能（50ms）
window_size = int(bgm.frame_rate * window_ms / 1000)

# -----------------------------
# 短音を鳴らすタイミングリスト作成
# -----------------------------
timings = []
cooldown_ms = 300  # 一度短音を鳴らしたら150msはスキップ
last_hit = -cooldown_ms
prev_peak = 0.0

for i in range(0, len(bgm_samples), window_size):
    window = bgm_samples[i:i+window_size]
    if len(window) == 0:
        continue

    peak = np.max(np.abs(window))
    ms_position = int(i / bgm.frame_rate * 1000)

    # 閾値を超え、かつ前のピークがある程度以上のときのみ短音を鳴らす
    if prev_peak > 0.05 and peak - prev_peak > 0.1 and ms_position - last_hit >= cooldown_ms:
        timings.append((ms_position, peak))
        last_hit = ms_position

    prev_peak = peak

# -----------------------------
# BGMに短音を重ねる
# -----------------------------
output = bgm
for t, peak in timings:
    # BGMのピークに応じて短音の音量を変える
    gain_db = 20 * np.log10(peak + 1e-6) + 6  # +6dBで少し大きめに
    short_adjusted = short_sound + gain_db
    output = output.overlay(short_adjusted, position=t)

# -----------------------------
# 保存
# -----------------------------
output.export("mixed_auto_filtered.wav", format="wav")
print("自動生成完了。mixed_auto_filtered.wav に保存されました。")
