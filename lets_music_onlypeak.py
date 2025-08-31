from pydub import AudioSegment
import numpy as np

# -----------------------------
# 音源読み込み
# -----------------------------
# output-kankyousubete
# bgm = AudioSegment.from_file("./output-natsu-bed.wav")       # 長めのBGM
bgm = AudioSegment.from_file("./output-kankyousubete.wav")       # 長めのBGM

short_sound = AudioSegment.from_file("./鍵を開ける1.mp3")       # 短い音

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
cooldown_ms = 1000  # 一度短音を鳴らしたら300msスキップ
last_hit = -cooldown_ms

for i in range(0, len(bgm_samples), window_size):
    window = bgm_samples[i:i+window_size]
    if len(window) == 0:
        continue

    peak = np.max(np.abs(window))
    ms_position = int(i / bgm.frame_rate * 1000)

    # 閾値を超え、かつクールダウンを考慮して短音を鳴らす
    if peak > 0.7 and ms_position - last_hit >= cooldown_ms:
        timings.append((ms_position, peak))
        last_hit = ms_position

# -----------------------------
# BGMに短音を重ねる
# -----------------------------
output = bgm
for t, peak in timings:
    # BGMのピークに応じて短音の音量を変える
    gain_db = 20 * np.log10(peak + 1e-6) + 6
    short_adjusted = short_sound + gain_db
    output = output.overlay(short_adjusted, position=t)

# -----------------------------
# 保存
# -----------------------------
output.export("mixed_auto_peak_only.wav", format="wav")
print("自動生成完了。mixed_auto_peak_only.wav に保存されました。")
