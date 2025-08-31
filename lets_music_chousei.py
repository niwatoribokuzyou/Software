from pydub import AudioSegment
import numpy as np

# -----------------------------
# 音源読み込み
# -----------------------------output-natsu-bed
bgm = AudioSegment.from_file("./output-natsu-bed.wav")       

# bgm = AudioSegment.from_file("./output-kankyousubete.wav")       
short_sound = AudioSegment.from_file("./鍵を開ける1.mp3")       

# -----------------------------
# BGMをnumpy配列に変換（分析用）
# -----------------------------
bgm_samples = np.array(bgm.get_array_of_samples(), dtype=np.float32)
bgm_samples /= np.max(np.abs(bgm_samples))  

window_ms = 50  
window_size = int(bgm.frame_rate * window_ms / 1000)

# -----------------------------
# 自動閾値とクールダウン計算
# -----------------------------
all_peaks = [np.max(np.abs(bgm_samples[i:i+window_size])) 
             for i in range(0, len(bgm_samples), window_size)]
mean_peak = np.mean(all_peaks)
std_peak = np.std(all_peaks)

threshold = mean_peak + 0.5 * std_peak  

# 基本値を大きめにして、BGMに応じて調整
min_cooldown = 300  # 最低300ms
num_peaks = np.count_nonzero(np.array(all_peaks) > threshold)
if num_peaks > 0:
    avg_peak_interval_ms = len(bgm) / num_peaks
    cooldown_ms = max(min_cooldown, int(avg_peak_interval_ms * 0.5))
else:
    cooldown_ms = min_cooldown

print(f"自動設定: threshold={threshold:.3f}, cooldown_ms={cooldown_ms}")

# -----------------------------
# 短音タイミング作成
# -----------------------------
timings = []
last_hit = -cooldown_ms

for i in range(0, len(bgm_samples), window_size):
    window = bgm_samples[i:i+window_size]
    if len(window) == 0:
        continue

    peak = np.max(np.abs(window))
    ms_position = int(i / bgm.frame_rate * 1000)

    if peak > threshold and ms_position - last_hit >= cooldown_ms:
        timings.append((ms_position, peak))
        last_hit = ms_position

# -----------------------------
# BGMに短音を重ねる
# -----------------------------
output = bgm
for t, peak in timings:
    gain_db = 20 * np.log10(peak + 1e-6) + 6
    short_adjusted = short_sound + gain_db
    output = output.overlay(short_adjusted, position=t)

# -----------------------------
# 保存
# -----------------------------
output.export("mixed_auto_adaptive_large_cd.wav", format="wav")
print("自動生成完了。mixed_auto_adaptive_large_cd.wav に保存されました。")
