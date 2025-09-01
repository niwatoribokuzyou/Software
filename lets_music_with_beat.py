import os
from io import BytesIO

import librosa
import numpy as np
from pydub import AudioSegment

from audio_events import AudioEvent, detect_and_slice


def blend_soundscape_music(music: bytes, soundscape: bytes) -> bytes:
    bio_music = BytesIO(music)
    music_segment = AudioSegment.from_file(bio_music, format="wav")
    music_segment.set_frame_rate(16000)

    total_len = len(music_segment)  # ms
    print(total_len)
    segment_len = total_len // 4  # 4分割
    min_interval = 800  # ms

    # 環境音を重ねるセグメント（2番目と4番目）
    env_segments = [1, 3]

    # 出力はBGMで初期化
    output = music_segment

    # librosaでビート解析
    y, sr = load_audio_bytes_anyformat(music, target_sr=16000, mono=True, fmt="wav")
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

    return output


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
    wav_music = "output-ribingu.wav"
    wav_env = "voice.wav"
    with open(wav_music, "rb") as f:
        music_data = f.read()
    with open(wav_env, "rb") as f:
        env_data = f.read()

    output = blend_soundscape_music(music_data, env_data)

    # 保存
    output_path = os.path.join("./outputs/", f"answer-{wav_env}")
    output.export(output_path, format="wav")
    print(f"環境音をBGM音量に合わせて2番目と4番目に重ねたBGMを保存しました: {output_path}")
