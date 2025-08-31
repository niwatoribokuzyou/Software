#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio_events.py — 生活音のイベント検出→切り出しをバッチ返却で提供
  - HPSSで過渡成分を強調 → オンセット → オフセット推定
  - （任意）類似間引き（DBSCAN）／ターゲット件数へのFacility-Location選抜
  - 戻り値は List[AudioEvent]（埋め込みは含めない）
  - 保存は save_events(...) に切り出し

依存:
  pip install librosa soundfile numpy pandas
  # 類似間引きや選抜を使う場合のみ:
  pip install scikit-learn
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.cluster import DBSCAN

# =========================
# データ構造
# =========================


@dataclass
class AudioEvent:
    id: str
    source_uri: str
    sr: int
    start_s: float
    end_s: float
    audio: np.ndarray  # float32, shape=(N,) or (N,C)
    onset_strength: float
    peak_rms: float
    tags: Optional[List[str]] = None
    feats: Optional[Dict[str, float]] = None
    extra: Optional[Dict[str, Any]] = None

    @property
    def duration_s(self) -> float:
        return float(self.end_s - self.start_s)

    def to_dict(self, with_audio: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        if not with_audio:
            d.pop("audio", None)
        return d


# =========================
# コア処理（HPSS / 検出）
# =========================


def hpss_percussive(
    y: np.ndarray,
    sr: int,
    *,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    kernel_time: int = 31,
    kernel_freq: int = 31,
    power: float = 2.0,
    margin_h: float = 1.0,
    margin_p: float = 1.0,
) -> np.ndarray:
    """STFTベースHPSSでパーカッシブ成分のみ再合成"""
    S_complex = librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, win_length=win_length, center=True
    )
    S_mag = np.abs(S_complex) + 1e-10
    H, P = librosa.decompose.hpss(
        S_mag,
        kernel_size=(kernel_time, kernel_freq),
        power=power,
        margin=(margin_h, margin_p),
    )
    denom = H + P + 1e-10
    mask_p = P / denom
    Yp = mask_p * S_complex
    yp = librosa.istft(Yp, hop_length=hop_length, win_length=win_length, length=len(y))
    return yp.astype(np.float32)


def detect_onsets(
    y_p: np.ndarray,
    sr: int,
    hop_length: int,
    *,
    backtrack: bool = True,
    delta: float = 0.2,
    wait_ms: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """オンセット包絡と検出フレーム（librosa 0.10以降: unitsは文字列必須 → frames固定）"""
    oenv = librosa.onset.onset_strength(
        y=y_p, sr=sr, hop_length=hop_length, aggregate=np.median
    )
    wait = max(1, int((wait_ms / 1000.0) * sr / hop_length))
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=oenv,
        sr=sr,
        hop_length=hop_length,
        backtrack=backtrack,
        delta=delta,
        wait=wait,
        units="frames",
    )
    return oenv, onset_frames


@dataclass
class _DetectedEvent:  # 内部用（オーディオ切り出し前）
    start_s: float
    end_s: float
    onset_strength: float
    peak_rms: float


def infer_offsets(
    y_p: np.ndarray,
    sr: int,
    hop_length: int,
    win_length: int,
    *,
    onset_frames: np.ndarray,
    min_dur: float = 0.05,
    max_dur: float = 1.0,
    alpha: float = 0.2,
    hang_frames: int = 5,
) -> List[_DetectedEvent]:
    """
    オフセット = オンセット後、RMS が (alpha * 局所ピーク) かつノイズフロアを下回る状態が
    hang_frames 連続した最初の点。最低/最長長も守る。
    """
    rms = librosa.feature.rms(
        y=y_p, frame_length=win_length, hop_length=hop_length, center=True
    ).flatten()
    t_total = len(y_p) / sr
    n_frames = len(rms)
    min_f = max(1, int(min_dur * sr / hop_length))
    max_f = max(min_f + 1, int(max_dur * sr / hop_length))
    noise_floor = np.percentile(rms, 20)

    events: List[_DetectedEvent] = []
    for i, on_f in enumerate(onset_frames):
        on_f = int(on_f)
        if on_f >= n_frames - 1:
            continue
        next_on_f = (
            int(onset_frames[i + 1]) if (i + 1) < len(onset_frames) else n_frames - 1
        )
        search_end = min(on_f + max_f, next_on_f)
        search_start = on_f
        if search_end <= search_start + min_f:
            continue

        segment_rms = rms[search_start:search_end]
        if len(segment_rms) == 0:
            continue
        peak = float(np.max(segment_rms))
        thr = max(noise_floor * 1.2, alpha * peak)

        off_f = None
        for f in range(search_start + min_f, search_end - hang_frames):
            window = rms[f : f + hang_frames]
            if np.all(window <= thr):
                off_f = f
                break
        if off_f is None:
            off_f = search_end

        start_s = float(librosa.frames_to_time(on_f, sr=sr, hop_length=hop_length))
        end_s = float(librosa.frames_to_time(off_f, sr=sr, hop_length=hop_length))
        end_s = min(end_s, t_total)
        onset_strength_est = float(
            np.clip(segment_rms[: min(len(segment_rms), min_f)].max(), 0.0, np.inf)
        )
        events.append(_DetectedEvent(start_s, end_s, onset_strength_est, peak))

    # 簡易クレンジング
    cleaned: List[_DetectedEvent] = []
    last_end = -1.0
    for ev in events:
        if ev.end_s <= ev.start_s:
            continue
        if last_end >= 0 and ev.start_s < last_end:
            continue
        cleaned.append(ev)
        last_end = ev.end_s
    return cleaned


# =========================
# 類似間引き（任意）
# =========================


def _event_feature_vector(seg: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """軽量特徴（log-Mel平均・分散＋スペクトル記述＋包絡）"""
    if len(seg) < int(0.02 * sr):
        return None
    S = librosa.feature.melspectrogram(
        y=seg, sr=sr, n_fft=1024, hop_length=256, n_mels=64, fmax=sr / 2
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    m_mu = S_db.mean(axis=1)
    m_std = S_db.std(axis=1)
    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr).mean()
    bw = librosa.feature.spectral_bandwidth(y=seg, sr=sr).mean()
    roll = librosa.feature.spectral_rolloff(y=seg, sr=sr, roll_percent=0.85).mean()
    flat = librosa.feature.spectral_flatness(y=seg).mean()
    zcr = librosa.feature.zero_crossing_rate(seg).mean()
    rms = float(np.sqrt(np.mean(seg ** 2)) + 1e-12)
    peak = float(np.max(np.abs(seg)) + 1e-12)
    crest = peak / rms
    feat = np.concatenate(
        [m_mu, m_std, np.array([centroid, bw, roll, flat, zcr, crest])]
    ).astype(np.float32)
    return feat


def nms_time(
    events: List[_DetectedEvent], window_ms: float = 50.0
) -> List[_DetectedEvent]:
    """時間的に近い重複を単純抑制（高スコア優先）"""
    if window_ms <= 0 or not events:
        return events
    w = window_ms / 1000.0
    cand = sorted(events, key=lambda e: (e.onset_strength, e.peak_rms), reverse=True)
    kept: List[_DetectedEvent] = []
    for ev in cand:
        if all(abs(ev.start_s - kv.start_s) > w for kv in kept):
            kept.append(ev)
    kept.sort(key=lambda e: e.start_s)
    return kept


def dedupe_events_dbscan(
    events: List[_DetectedEvent],
    y_feat: np.ndarray,
    sr: int,
    *,
    eps_cosine: float = 0.35,
    min_samples: int = 2,
) -> List[_DetectedEvent]:
    """DBSCANで類似イベントをマージ（代表=onset_strength高）"""
    if not events:
        return events
    feats, idx = [], []
    for i, ev in enumerate(events):
        s = int(ev.start_s * sr)
        e = int(ev.end_s * sr)
        s = max(0, s)
        e = min(len(y_feat), e)
        if e <= s:
            continue
        f = _event_feature_vector(y_feat[s:e], sr)
        if f is None or not np.all(np.isfinite(f)):
            continue
        feats.append(f)
        idx.append(i)
    if not feats:
        return events

    X = np.vstack(feats).astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    labels = DBSCAN(
        eps=eps_cosine, min_samples=min_samples, metric="cosine"
    ).fit_predict(Xn)

    # 代表を選抜
    best: Dict[int, Tuple[Tuple[float, float], _DetectedEvent]] = {}
    singles: List[_DetectedEvent] = []
    for j, original_idx in enumerate(idx):
        ev = events[original_idx]
        lbl = labels[j]
        score = (ev.onset_strength, ev.peak_rms)
        if lbl == -1:
            singles.append(ev)
        else:
            if (lbl not in best) or (score > best[lbl][0]):
                best[lbl] = (score, ev)
    kept = singles + [v[1] for v in best.values()]
    kept.sort(key=lambda e: e.start_s)
    return kept


# =========================
# Facility-Location でK件選抜（任意）
# =========================


def _build_event_features(events: List[_DetectedEvent], y_feat: np.ndarray, sr: int):
    """選抜用の特徴X・品質q・時刻t・元インデックスidxを構成"""
    feats, qs, ts, idxs = [], [], [], []
    onset_list = [ev.onset_strength for ev in events]
    peak_list = [ev.peak_rms for ev in events]
    o_min, o_max = float(np.min(onset_list)), float(np.max(onset_list))
    p_min, p_max = float(np.min(peak_list)), float(np.max(peak_list))

    def _norm(x, a, b):
        return 0.0 if (b - a) < 1e-12 else float((x - a) / (b - a))

    for i, ev in enumerate(events):
        s = int(ev.start_s * sr)
        e = int(ev.end_s * sr)
        s = max(0, s)
        e = min(len(y_feat), e)
        if e - s < int(0.02 * sr):
            continue
        f = _event_feature_vector(y_feat[s:e], sr)
        if f is None or not np.all(np.isfinite(f)):
            continue
        feats.append(f)
        q = 0.5 * _norm(ev.onset_strength, o_min, o_max) + 0.5 * _norm(
            ev.peak_rms, p_min, p_max
        )
        qs.append(q)
        ts.append(ev.start_s)
        idxs.append(i)

    if not feats:
        return None, None, None, None
    X = np.vstack(feats).astype(np.float32)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    q = np.asarray(qs, dtype=np.float32)
    t = np.asarray(ts, dtype=np.float32)
    idx = np.asarray(idxs, dtype=np.int32)
    return Xn, q, t, idx


def facility_location_select(
    events: List[_DetectedEvent],
    y_feat: np.ndarray,
    sr: int,
    *,
    k: int,
    beta: float = 0.2,
    time_ms: float = 0.0,
) -> List[_DetectedEvent]:
    """
    F(S) = Σ_i max_{j∈S} s_ij + β Σ_{j∈S} q_j を貪欲最大化して K件選抜
      - s_ij: コサイン類似度（[0,1]射影）
      - q_j : イベント品質（正規化）
      - time_ms > 0: 時間差ガウシアンで類似度減衰
    """
    Xn, q, t, idx = _build_event_features(events, y_feat, sr)
    if Xn is None:
        # 特徴が作れない場合は強い順にK件
        k = min(k, len(events))
        return sorted(
            events, key=lambda e: (e.onset_strength, e.peak_rms), reverse=True
        )[:k]

    N = Xn.shape[0]
    k = int(min(max(1, k), N))

    S = Xn @ Xn.T
    S = 0.5 * (S + 1.0)
    np.fill_diagonal(S, 1.0)

    if time_ms and time_ms > 0:
        sigma = float(time_ms) / 1000.0
        dt = np.abs(t[:, None] - t[None, :])
        Wt = np.exp(-0.5 * (dt / sigma) ** 2)
        S = S * Wt

    selected_local: List[int] = []
    not_sel = np.ones(N, dtype=bool)
    cover = np.zeros(N, dtype=np.float32)

    for _ in range(k):
        D = S - cover[:, None]
        np.maximum(D, 0.0, out=D)
        D[:, ~not_sel] = 0.0
        gains_cov = D.sum(axis=0)
        gains = gains_cov + beta * q
        gains[~not_sel] = -np.inf
        j = int(np.argmax(gains))
        if not np.isfinite(gains[j]):
            break
        selected_local.append(j)
        not_sel[j] = False
        cover = np.maximum(cover, S[:, j])

    chosen_idx = idx[np.array(selected_local, dtype=int)].tolist()
    chosen = [events[i] for i in chosen_idx]
    chosen.sort(key=lambda e: e.start_s)
    return chosen


# =========================
# メインAPI（バッチ返却）
# =========================


def detect_and_slice(
    input_audio: Union[str, Tuple[np.ndarray, int]],
    *,
    sr: int = 16000,
    n_fft: int = 1024,
    hop: int = 256,
    win: int = 1024,
    harm_kernel: int = 31,
    perc_kernel: int = 31,
    hpss_power: float = 2.0,
    margin_h: float = 1.0,
    margin_p: float = 1.0,
    delta: float = 0.2,
    wait_ms: float = 50.0,
    backtrack: bool = True,
    min_dur: float = 0.05,
    max_dur: float = 1.0,
    alpha: float = 0.2,
    hang: int = 5,
    # 以降は任意機能
    time_nms_ms: float = 0.0,
    dedupe: bool = False,
    dedupe_eps: float = 0.35,
    dedupe_min_samples: int = 2,
    target_k: int = 0,
    select_beta: float = 0.2,
    select_time_ms: float = 0.0,
) -> List[AudioEvent]:
    """
    入力: Wavパス or (y, sr) タプル
    出力: List[AudioEvent]（保存はしない）
    """
    # 入力処理
    if isinstance(input_audio, tuple):
        y, sr_in = input_audio
        if sr_in != sr:
            y = librosa.resample(y.astype(np.float32), orig_sr=sr_in, target_sr=sr)
    else:
        y, _ = librosa.load(str(input_audio), sr=sr, mono=True)
    y = librosa.util.normalize(y.astype(np.float32))
    source_uri = input_audio[0] if isinstance(input_audio, tuple) else str(input_audio)

    # HPSS → オンセット → オフセット
    y_p = hpss_percussive(
        y,
        sr,
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        kernel_time=harm_kernel,
        kernel_freq=perc_kernel,
        power=hpss_power,
        margin_h=margin_h,
        margin_p=margin_p,
    )
    _, onset_frames = detect_onsets(
        y_p, sr, hop, backtrack=backtrack, delta=delta, wait_ms=wait_ms
    )
    det = infer_offsets(
        y_p,
        sr,
        hop,
        win,
        onset_frames=onset_frames,
        min_dur=min_dur,
        max_dur=max_dur,
        alpha=alpha,
        hang_frames=hang,
    )

    # 任意の後処理
    if time_nms_ms > 0:
        det = nms_time(det, window_ms=time_nms_ms)
    if dedupe:
        det = dedupe_events_dbscan(
            det,
            y_feat=y_p,
            sr=sr,
            eps_cosine=dedupe_eps,
            min_samples=dedupe_min_samples,
        )
    if target_k and target_k > 0 and len(det) > target_k:
        det = facility_location_select(
            det, y_feat=y_p, sr=sr, k=target_k, beta=select_beta, time_ms=select_time_ms
        )

    # AudioEvent に変換
    events: List[AudioEvent] = []
    for ev in det:
        s = int(max(0, round(ev.start_s * sr)))
        e = int(min(len(y), round(ev.end_s * sr)))
        if e - s < int(0.01 * sr):  # 10ms未満はスキップ
            continue
        seg = y[s:e].astype(np.float32, copy=False)
        events.append(
            AudioEvent(
                id=str(uuid.uuid4()),
                source_uri=source_uri,
                sr=sr,
                start_s=ev.start_s,
                end_s=ev.end_s,
                audio=seg,
                onset_strength=ev.onset_strength,
                peak_rms=ev.peak_rms,
            )
        )
    return events


# =========================
# 保存ユーティリティ（別関数）
# =========================


def events_to_dataframe(events: List[AudioEvent]) -> pd.DataFrame:
    rows = []
    for e in events:
        rows.append(
            {
                "id": e.id,
                "source_uri": e.source_uri,
                "sr": e.sr,
                "start_s": e.start_s,
                "end_s": e.end_s,
                "duration_s": e.duration_s,
                "onset_strength": e.onset_strength,
                "peak_rms": e.peak_rms,
                "tags": ",".join(e.tags) if e.tags else "",
            }
        )
    return pd.DataFrame(rows)


def save_events(
    events: List[AudioEvent],
    out_dir: str,
    *,
    save_audio: bool = True,
    subtype: str = "PCM_16",
    write_csv: bool = True,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    イベントをディスクに保存（音は id.wav、メタは events.csv と meta.json）
    """
    os.makedirs(out_dir, exist_ok=True)
    if save_audio:
        for e in events:
            sf.write(
                os.path.join(out_dir, f"{e.id}.wav"), e.audio, e.sr, subtype=subtype
            )
    if write_csv:
        df = events_to_dataframe(events)
        df.to_csv(os.path.join(out_dir, "events.csv"), index=False)
    if meta is not None:
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # --- Debug CLI: detect -> dump to folder ---------------------------------
    """
    example:
    python audio_events.py voice.wav --out_dir ./events \
      --dedupe --dedupe_eps 0.0005 --dedupe_min_samples 2 \
      --target_k 10 --select_beta 0.3 --select_time_ms 0
    """
    import argparse
    import datetime as _dt
    import glob

    ap = argparse.ArgumentParser(
        description="Debug mode: detect events and dump WAV/CSV into the specified directory."
    )
    ap.add_argument("input", help="入力WAVファイル or ディレクトリ")
    ap.add_argument("--pattern", default=".wav", help="ディレクトリ入力時の拡張子フィルタ（例: .wav）")
    ap.add_argument("--out_dir", default="./events_debug", help="書き出し先ルートディレクトリ")
    ap.add_argument("--sr", type=int, default=16000, help="処理サンプルレート")

    # STFT/HPSS
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--win", type=int, default=1024)
    ap.add_argument("--harm_kernel", type=int, default=31)
    ap.add_argument("--perc_kernel", type=int, default=31)
    ap.add_argument("--hpss_power", type=float, default=2.0)
    ap.add_argument("--margin_h", type=float, default=1.0)
    ap.add_argument("--margin_p", type=float, default=1.0)

    # Onset/Boundary
    ap.add_argument("--delta", type=float, default=0.2)
    ap.add_argument("--wait_ms", type=float, default=50.0)
    ap.add_argument("--no_backtrack", action="store_true")
    ap.add_argument("--min_dur", type=float, default=0.05)
    ap.add_argument("--max_dur", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--hang", type=int, default=5)

    # Optional post-process
    ap.add_argument("--time_nms_ms", type=float, default=0.0, help="時間NMSの窓 [ms]（0で無効）")
    ap.add_argument("--dedupe", action="store_true", help="類似間引きを有効化（DBSCAN）")
    ap.add_argument(
        "--dedupe_eps", type=float, default=0.35, help="DBSCAN eps（cosine距離）"
    )
    ap.add_argument("--dedupe_min_samples", type=int, default=2)
    ap.add_argument(
        "--target_k", type=int, default=0, help=">0でFacility-LocationによりK件へ圧縮"
    )
    ap.add_argument(
        "--select_beta", type=float, default=0.2, help="Facility-Locationの品質重みβ"
    )
    ap.add_argument(
        "--select_time_ms", type=float, default=0.0, help="類似度の時間減衰スケール[ms]（0で無効）"
    )

    # Output options
    ap.add_argument("--save_csv_only", action="store_true", help="音声を保存せずCSVのみ出力")
    ap.add_argument("--subtype", default="PCM_16", help="WAV subtype（例: PCM_16, FLOAT）")

    args = ap.parse_args()

    # 入力の解決
    if os.path.isdir(args.input):
        paths = sorted(
            p
            for p in glob.glob(os.path.join(args.input, "*"))
            if p.lower().endswith(args.pattern.lower())
        )
    else:
        paths = [args.input]

    if len(paths) == 0:
        print("[WARN] 入力が見つかりませんでした。")
        raise SystemExit(1)

    # ルート出力ディレクトリ
    os.makedirs(args.out_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[DEBUG] Start dump -> {args.out_dir}  ({ts})")

    for path in paths:
        print(f"[INFO] Processing: {path}")
        events = detect_and_slice(
            path,
            sr=args.sr,
            n_fft=args.n_fft,
            hop=args.hop,
            win=args.win,
            harm_kernel=args.harm_kernel,
            perc_kernel=args.perc_kernel,
            hpss_power=args.hpss_power,
            margin_h=args.margin_h,
            margin_p=args.margin_p,
            delta=args.delta,
            wait_ms=args.wait_ms,
            backtrack=(not args.no_backtrack),
            min_dur=args.min_dur,
            max_dur=args.max_dur,
            alpha=args.alpha,
            hang=args.hang,
            time_nms_ms=args.time_nms_ms,
            dedupe=args.dedupe,
            dedupe_eps=args.dedupe_eps,
            dedupe_min_samples=args.dedupe_min_samples,
            target_k=args.target_k,
            select_beta=args.select_beta,
            select_time_ms=args.select_time_ms,
        )
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(args.out_dir, stem)
        meta = {
            "source": os.path.abspath(path),
            "sr": args.sr,
            "params": {
                "n_fft": args.n_fft,
                "hop": args.hop,
                "win": args.win,
                "harm_kernel": args.harm_kernel,
                "perc_kernel": args.perc_kernel,
                "hpss_power": args.hpss_power,
                "margin_h": args.margin_h,
                "margin_p": args.margin_p,
                "delta": args.delta,
                "wait_ms": args.wait_ms,
                "backtrack": (not args.no_backtrack),
                "min_dur": args.min_dur,
                "max_dur": args.max_dur,
                "alpha": args.alpha,
                "hang": args.hang,
                "time_nms_ms": args.time_nms_ms,
                "dedupe": args.dedupe,
                "dedupe_eps": args.dedupe_eps,
                "dedupe_min_samples": args.dedupe_min_samples,
                "target_k": args.target_k,
                "select_beta": args.select_beta,
                "select_time_ms": args.select_time_ms,
            },
            "timestamp": ts,
        }
        save_events(
            events,
            out_dir=out,
            save_audio=(not args.save_csv_only),
            subtype=args.subtype,
            write_csv=True,
            meta=meta,
        )
        print(f"[OK] Saved {len(events)} events -> {out}")
