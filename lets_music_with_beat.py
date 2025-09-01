
# # from pydub import AudioSegment
# # import librosa
# # import os
# # import numpy as np

# # wav_files = ["output-ribingu.wav", "output-bed.wav"]
# # min_interval = 800  # ms

# # for wav_file in wav_files:
# #     path = wav_file
# #     print("path", path)
    
# #     # BGM読み込み（全体流す）
# #     bgm = AudioSegment.from_file(path)
# #     total_len = len(bgm)  # ms
# #     segment_len = total_len // 4  # 4分割

# #     # 短音読み込み（音量は関係なし）
# #     short_sound_orig = AudioSegment.from_file("./鍵を開ける1.mp3")
# #     short_sound_orig = short_sound_orig - short_sound_orig.dBFS  # 基準化

# #     # 環境音を重ねるセグメント（2番目と4番目）
# #     env_segments = [1, 3]

# #     # 出力はBGMで初期化
# #     output = bgm

# #     # librosaでビート解析
# #     y, sr = librosa.load(path)
# #     print("y",y)
# #     print("type:", type(y))          # <class 'numpy.ndarray'>
# #     print("dtype:", y.dtype)         # float32
# #     print("shape:", y.shape) 
# #     exit()
# #     tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
# #     beat_times = librosa.frames_to_time(beat_frames, sr=sr) * 1000
# #     beat_times = beat_times.astype(int)

# #     last_time = -min_interval

# #     # 環境音を重ねる
# #     for seg_idx in env_segments:
# #         start_ms = seg_idx * segment_len
# #         end_ms = (seg_idx + 1) * segment_len if seg_idx < 3 else total_len

# #         for t in beat_times:
# #             if start_ms <= t < end_ms and t - last_time >= min_interval:
# #                 # 短音の音量をBGMのその位置の音量に合わせる
# #                 bgm_slice = bgm[t:t+len(short_sound_orig)]
# #                 gain = bgm_slice.dBFS
# #                 short_sound = short_sound_orig.apply_gain(gain+10)

# #                 output = output.overlay(short_sound, position=int(t))
# #                 last_time = t

# #     # 保存
# #     output_path = os.path.join('./test/', f'answer-{wav_file}')
# #     output.export(output_path, format="wav")
# #     print(f"環境音をBGM音量に合わせて2番目と4番目に重ねたBGMを保存しました: {output_path}")




# from pydub import AudioSegment
# import librosa
# import os
# import numpy as np
# import soundfile as sf

# def rms_db(wave):
#     rms = np.sqrt(np.mean(wave**2))
#     return 20 * np.log10(rms + 1e-9)

# # -----------------------------
# # 短音をnumpyでBGMに重ねる
# # -----------------------------
# def overlay_short_sounds_numpy(BGM, sr, short_audio, output_path):
#     """
#     BGM: np.ndarray, float32 [-1,1]
#     sr: サンプルレート
#     short_sound_path: 短音ファイルパス
#     output_path: 保存先
#     min_interval: ms, 短音を重ねる最小間隔
#     gain_db: BGMよりどれくらい大きくするか
#     """
#     # 短音読み込み
#     min_interval=800
#     gain_db=2.0
#     # short_audio, short_sr = librosa.load(short_sound_path, sr=sr)

    

#     # print("short_audio", short_audio)
#     # print("short_audio", type(short_audio))
#     # print("dtype:", short_audio.dtype)
#     # print("shape:", short_audio.shape)
#     # exit()

#     output = BGM.copy()
#     total_len = len(BGM)
#     segment_len = total_len // 4
#     env_segments = [1, 3]  # 2番目と4番目のセグメントのみ短音を重ねる

#     # ビート解析
#     tempo, beat_frames = librosa.beat.beat_track(y=BGM, sr=sr)
#     beat_times = librosa.frames_to_time(beat_frames, sr=sr)  # 秒
#     beat_samples = (beat_times * sr).astype(int)

#     last_sample = -int(min_interval / 1000 * sr)

#     for t in beat_samples:
#         seg_idx = t // segment_len
#         if seg_idx not in env_segments:
#             continue  # 環境音を重ねないターンはスキップ

#         if t - last_sample >= int(min_interval / 1000 * sr):
#             start = t
#             end = t + len(short_audio)
#             if end > len(output):
#                 end = len(output)
#                 short_slice = short_audio[:end-start]
#             else:
#                 short_slice = short_audio

#             # BGM の RMS を取得して短音をスケーリング
#             bgm_rms_db = rms_db(output[start:end])
#             short_rms_db = rms_db(short_slice)
#             scale_db = bgm_rms_db - short_rms_db + gain_db
#             scale = 10 ** (scale_db / 20)
#             short_slice_scaled = short_slice * scale

#             # 加算してクリッピング
#             output[start:end] += short_slice_scaled
#             output[start:end] = np.clip(output[start:end], -1.0, 1.0)

#             last_sample = t

#     # 保存
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)
#     sf.write(output_path, output, sr)
#     print(f"✅ 保存しました: {output_path}")
#     return output








import os
from io import BytesIO

import librosa
import numpy as np
from pydub import AudioSegment

from audio_events import detect_and_slice


# 生成AIが作った音楽と環境音をmp3のバイナリで投げるといい感じにくっつけたmp3のバイナリを返してくれる関数
def blend_soundscape_music(music: bytes, soundscape: bytes) -> bytes:
    bio_music = BytesIO(music)
    music_segment = AudioSegment.from_file(bio_music, format="mp3")
    music_segment.set_frame_rate(16000)

    total_len = len(music_segment)  # ms
    segment_len = total_len // 4  # 4分割
    min_interval = 800  # ms

    # 環境音を重ねるセグメント（2番目と4番目）
    env_segments = [1, 3]

    # 出力はBGMで初期化
    output = music_segment

    # librosaでビート解析
    y, sr = load_audio_bytes_anyformat(music, target_sr=16000, mono=True, fmt="mp3")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=16000)
    beat_times = librosa.frames_to_time(beat_frames, sr=16000) * 1000
    beat_times = beat_times.astype(int)

    # 本当はもっと減るはずだが面倒なのでとりあえず
    num_beats = len(beat_frames)

    env_events = detect_and_slice(
        soundscape,
        sr=16000,
        delta=0.2,
        dedupe=True,
        dedupe_eps=0.0001,
        target_k=num_beats,
        select_time_ms=0,
        select_beta=0.2,
    )

    last_time = -min_interval

    # 環境音を重ねる
    counter = 0
    for seg_idx in env_segments:
        start_ms = seg_idx * segment_len
        end_ms = (seg_idx + 1) * segment_len if seg_idx < 3 else total_len
        # print(start_ms)

        for t in beat_times:
            if start_ms <= t < end_ms and t - last_time >= min_interval:
                # 短音の音量をBGMのその位置の音量に合わせる
                short_sound_orig = env_events[counter].audio
                short_sound_orig /= np.max(np.abs(short_sound_orig))  # 正規化
                short_sound_seg = np_float32_to_audiosegment(
                    short_sound_orig, sr=16000, sample_width=2
                )
                music_slice = music_segment[t : t + len(short_sound_seg)]
                gain = music_slice.dBFS
                short_sound = short_sound_seg.apply_gain(gain + 20)

                output = output.overlay(short_sound, position=int(t))
                last_time = t
                counter += 1

    mp3_binary_data = BytesIO()
    output.export(mp3_binary_data, format="mp3")
    binary_mp3 = mp3_binary_data.getvalue()

    return binary_mp3


def load_audio_bytes_anyformat(
    data: bytes, target_sr: int | None = None, mono: bool = True, fmt: str | None = None
):
    """
    任意フォーマット(bytes)を y(float32, -1..1), sr で返す。
    WAV/FLAC/OGGはlibrosa/soundfileで直接、MP3/M4Aなどはpydub経由。
    """
    bio = BytesIO(data)

    # まずは soundfile 経由（WAV/FLAC/OGG）をトライ
    try:
        # soundfile が読める形式ならこれでOK
        y, sr = librosa.load(bio, sr=target_sr, mono=mono)
        return y.astype(np.float32), sr
    except Exception:
        pass  # だめなら pydub にフォールバック

    # pydub でデコード（MP3/M4A/Opusなど）
    bio.seek(0)
    # 例: fmt="mp3" / "m4a" / "wav"。不明ならNoneでも通ることが多い
    seg = AudioSegment.from_file(bio, format=fmt)
    sr = seg.frame_rate
    arr = np.array(seg.get_array_of_samples())

    # チャンネル整形（pydubは [L R L R ...] の並び）
    if seg.channels > 1:
        arr = arr.reshape((-1, seg.channels)).T  # (C, N)
        if mono:
            arr = arr.mean(axis=0)  # モノラル化
    # 正規化：int -> [-1,1]
    # 8bit:128, 16bit:32768, 24bit:8388608
    scale = float(1 << (8 * seg.sample_width - 1))
    y = arr.astype(np.float32) / scale

    if (target_sr is not None) and (target_sr != sr):
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return y, sr


def np_float32_to_audiosegment(
    y: np.ndarray, sr: int, sample_width: int = 2
) -> AudioSegment:
    """
    np.float32 波形 (-1..1) -> pydub.AudioSegment
    y: shape が (N,), (C,N), (N,C) を許容（C=チャンネル）
    sr: サンプリング周波数
    sample_width: 2=16bit, 4=32bit（整数PCM）。通常は 2 でOK。
    """
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    # 形状を (C, N) にそろえる
    if y.ndim == 1:
        y = y[np.newaxis, :]  # (1, N)
    elif y.ndim == 2:
        # どちらがチャンネル軸か推定（「次元が小さい方≦8」をチャンネルとみなす）
        if y.shape[0] <= 8 and y.shape[1] > 8:
            pass  # (C, N)
        elif y.shape[1] <= 8 and y.shape[0] > 8:
            y = y.T  # (N, C) -> (C, N)
        else:
            # あいまいなら先頭をチャンネルとみなす
            pass
    else:
        raise ValueError("y は 1D か 2D（(N,) / (C,N) / (N,C)）にしてください。")

    # クリップ（安全）
    y = np.clip(y, -1.0, 1.0)

    # 量子化（整数PCMへ）
    if sample_width == 2:
        # 16-bit signed PCM
        scale = 32767.0
        pcm = np.round(y * scale).astype("<i2")  # little-endian int16
    elif sample_width == 4:
        # 32-bit signed PCM
        scale = 2147483647.0
        pcm = np.round(y * scale).astype("<i4")  # little-endian int32
    else:
        raise ValueError("sample_width は 2（16bit）か 4（32bit int）を指定してください。")

    # (C, N) -> インターリーブ [L R L R ...]
    interleaved = pcm.transpose(1, 0).reshape(-1)  # (N, C) -> (N*C,)
    raw = interleaved.tobytes()

    seg = AudioSegment(
        data=raw,
        sample_width=sample_width,  # バイト数/サンプル
        frame_rate=int(sr),
        channels=int(pcm.shape[0]),
    )
    return seg


if __name__ == "__main__":
    music = "output-ribingu.mp3"
    env = "voice.mp3"
    with open(music, "rb") as f:
        music_data = f.read()
    with open(env, "rb") as f:
        env_data = f.read()

    output = blend_soundscape_music(music_data, env_data)

    # 保存
    output_path = os.path.join("./outputs/", f"answer-{env}")
    with open(output_path, "wb") as f:
        f.write(output)
    print(f"環境音をBGM音量に合わせて2番目と4番目に重ねたBGMを保存しました: {output_path}")