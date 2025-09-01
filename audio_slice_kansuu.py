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

from audio_events import detect_and_slice, save_events
import os
import datetime as _dt
import glob
from typing import Union, List

def process_audio_files(
    input_path: str,
    out_dir: str = "./events",
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
    time_nms_ms: float = 0.0,
    dedupe: bool = True,
    dedupe_eps: float = 0.0005,
    dedupe_min_samples: int = 2,
    target_k: int = 10,
    select_beta: float = 0.3,
    select_time_ms: float = 0.0,
    save_csv_only: bool = False,
    subtype: str = "PCM_16",
    pattern: str = ".wav"
) -> None:
    """音声ファイルまたはディレクトリを処理してイベントを抽出・保存"""
    
    # 入力の解決
    if os.path.isdir(input_path):
        paths = sorted(
            p
            for p in glob.glob(os.path.join(input_path, "*"))
            if p.lower().endswith(pattern.lower())
        )
    else:
        paths = [input_path]

    if len(paths) == 0:
        print("[WARN] 入力が見つかりませんでした。")
        return

    os.makedirs(out_dir, exist_ok=True)
    ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"[DEBUG] Start dump -> {out_dir}  ({ts})")

    for path in paths:
        print(f"[INFO] Processing: {path}")
        events = detect_and_slice(
            path,
            sr=sr,
            n_fft=n_fft,
            hop=hop,
            win=win,
            harm_kernel=harm_kernel,
            perc_kernel=perc_kernel,
            hpss_power=hpss_power,
            margin_h=margin_h,
            margin_p=margin_p,
            delta=delta,
            wait_ms=wait_ms,
            backtrack=backtrack,
            min_dur=min_dur,
            max_dur=max_dur,
            alpha=alpha,
            hang=hang,
            time_nms_ms=time_nms_ms,
            dedupe=dedupe,
            dedupe_eps=dedupe_eps,
            dedupe_min_samples=dedupe_min_samples,
            target_k=target_k,
            select_beta=select_beta,
            select_time_ms=select_time_ms,
        )
        stem = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(out_dir, stem)
        meta = {
            "source": os.path.abspath(path),
            "sr": sr,
            "params": locals(),
            "timestamp": ts,
        }

        # save_events(
        #     events,
        #     out_dir=out,
        #     save_audio=(not save_csv_only),
        #     subtype=subtype,
        #     write_csv=True,
        #     meta=meta,
        # )

        print("len(events)",len(events))
        print("short_audio", events[0].audio)
        print("short_audio", type(events[0].audio))
        print("dtype:", events[0].audio.dtype)
        print("shape:", events[0].audio.shape)
        print("終わり")
        audios = [events[i].audio for i in range(len(events))]
        return audios
        print(f"[OK] Saved {len(events)} events -> {out}")
        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAa")
        print("events", events[0].audio)