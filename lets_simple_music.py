from pydub import AudioSegment
import librosa

y, sr = librosa.load("output-kankyousubete.wav")
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
print(f"推定BPM: {tempo}")
# exit()
# -----------------------------
# 音源読み込み
# -----------------------------
bgm = AudioSegment.from_file("./output-kankyousubete.wav")       # 長めのBGM
short_sound = AudioSegment.from_file("./鍵を開ける1.mp3")   # 短い音

# -----------------------------
# BPM設定
# -----------------------------
# tempo = 80
# ms_per_beat = 60000 // int(tempo) # 1拍あたりのミリ秒

ms_per_beat = 60000 // int(tempo.item()) # 1拍あたりのミリ秒

# -----------------------------
# 短音を鳴らすタイミング作成
# -----------------------------
timings = []
current_time = 0
while current_time < len(bgm):
    timings.append(current_time)
    current_time += ms_per_beat  # BPMに基づき次の拍へ

# -----------------------------
# BGMに短音を重ねる
# -----------------------------
output = bgm
for t in timings:
    output = output.overlay(short_sound, position=int(t))

# -----------------------------
# 保存
# -----------------------------
output.export("mixed_bpm.wav", format="wav")
print("BPMに合わせた短音付きBGMを保存しました: mixed_bpm.wav")
