# from pydub import AudioSegment
# import librosa
# import os
# # -----------------------------
# # 音源読み込みとビート解析
# # -----------------------------
# # y, sr = librosa.load("output-natsu-bed.wav")
# # y, sr = librosa.load("output-ribingu-hiru.wav")
# # ./output-kankyousubete.wav"


# wav_files = ["output-ribingu.wav","output-bed.wav"]
# # wav_folder = "./generated_music/"
# # wav_files = os.listdir(wav_folder)


# for wav_file in wav_files:

#     # path = os.path.join(wav_folder, wav_file)
#     path = wav_file
#     print("path", path)
#     y, sr = librosa.load(path)
#     tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#     print(f"推定BPM: {tempo}")

#     # フレーム番号 → 時間（秒） → ミリ秒に変換
#     beat_times = librosa.frames_to_time(beat_frames, sr=sr) * 1000  # ms単位
#     beat_times = beat_times.astype(int)

#     # -----------------------------
#     # 音声ファイル読み込み
#     # -----------------------------
#     # bgm = AudioSegment.from_file("./output-natsu-bed.wav")
#     # bgm = AudioSegment.from_file("./output-ribingu-hiru.wav")

#     bgm = AudioSegment.from_file(path)
#     short_sound = AudioSegment.from_file("./鍵を開ける1.mp3")

#     # -----------------------------
#     # BGMに短音を重ねる
#     # -----------------------------
#     output = bgm
#     # for t in beat_times:
#     #     if t < len(bgm):  # BGMの長さを超えないように
#     #         output = output.overlay(short_sound, position=int(t))


#     min_interval = 800  # ms（←ここを変えると間隔調整できる）
#     last_time = -min_interval  # 最初に必ず鳴らせるように

#     for t in beat_times:
#         if t - last_time >= min_interval and t < len(bgm):
#             output = output.overlay(short_sound, position=int(t))
#             last_time = t  # 鳴らした時刻を更新

#     output_path = os.path.join('./test/', f'answer-{wav_file}')
#     # -----------------------------
#     # 保存
#     # -----------------------------
#     output.export(output_path, format="wav")
#     print(f"ビートに合わせた短音付きBGMを保存しました:{output_path}")





from pydub import AudioSegment
import librosa
import os
import numpy as np

wav_files = ["output-ribingu.wav", "output-bed.wav"]
min_interval = 800  # ms

for wav_file in wav_files:
    path = wav_file
    print("path", path)
    
    # BGM読み込み（全体流す）
    bgm = AudioSegment.from_file(path)
    total_len = len(bgm)  # ms
    segment_len = total_len // 4  # 4分割

    # 短音読み込み（音量は関係なし）
    short_sound_orig = AudioSegment.from_file("./鍵を開ける1.mp3")
    short_sound_orig = short_sound_orig - short_sound_orig.dBFS  # 基準化

    # 環境音を重ねるセグメント（2番目と4番目）
    env_segments = [1, 3]

    # 出力はBGMで初期化
    output = bgm

    # librosaでビート解析
    y, sr = librosa.load(path)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr) * 1000
    beat_times = beat_times.astype(int)

    last_time = -min_interval

    # 環境音を重ねる
    for seg_idx in env_segments:
        start_ms = seg_idx * segment_len
        end_ms = (seg_idx + 1) * segment_len if seg_idx < 3 else total_len

        for t in beat_times:
            if start_ms <= t < end_ms and t - last_time >= min_interval:
                # 短音の音量をBGMのその位置の音量に合わせる
                bgm_slice = bgm[t:t+len(short_sound_orig)]
                gain = bgm_slice.dBFS
                short_sound = short_sound_orig.apply_gain(gain+10)

                output = output.overlay(short_sound, position=int(t))
                last_time = t

    # 保存
    output_path = os.path.join('./test/', f'answer-{wav_file}')
    output.export(output_path, format="wav")
    print(f"環境音をBGM音量に合わせて2番目と4番目に重ねたBGMを保存しました: {output_path}")
