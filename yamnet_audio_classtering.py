import os
import glob
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# -------------------------
# 入力設定
# -------------------------
in_dir = "./snippets"  # 短音WAVフォルダ
pattern = "*.wav"
n_clusters = 5         # クラスタ数

# -------------------------
# YAMNetモデル読み込み (CPUでOK)
# -------------------------
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = yamnet_model.class_map_path().numpy()
class_names = [c.decode("utf-8") for c in tf.io.gfile.GFile(class_map_path)]

# -------------------------
# 音声処理関数
# -------------------------
def extract_embedding(path):
    # wavを読み込んで16kHzに変換
    wav, sr = librosa.load(path, sr=16000, mono=True)
    waveform = wav.astype(np.float32)

    # YAMNetで埋め込みとスコア取得
    scores, embeddings, _ = yamnet_model(waveform)
    # embeddings: (フレーム数, 1024)
    emb_mean = np.mean(embeddings.numpy(), axis=0)
    score_mean = np.mean(scores.numpy(), axis=0)
    return emb_mean, score_mean

# -------------------------
# 全音声ファイル処理
# -------------------------
paths = sorted(glob.glob(os.path.join(in_dir, pattern)))
features, probs = [], []

for path in paths:
    try:
        emb, score = extract_embedding(path)
        features.append(emb)
        probs.append(score)
    except Exception as e:
        print(f"[WARN] {path}: {e}")

features = np.array(features)
probs = np.array(probs)

# -------------------------
# 各音声のトップカテゴリ表示
# -------------------------
top_classes = [class_names[np.argmax(p)] for p in probs]
for pth, cls in zip(paths, top_classes):
    print(f"{os.path.basename(pth)} → {cls}")

# -------------------------
# クラスタリング
# -------------------------
km = KMeans(n_clusters=n_clusters, random_state=42)
labels = km.fit_predict(features)

# -------------------------
# PCAで可視化
# -------------------------
X2 = PCA(n_components=2).fit_transform(features)
plt.figure(figsize=(6,5))
for c in range(n_clusters):
    idx = labels == c
    plt.scatter(X2[idx,0], X2[idx,1], label=f"cluster {c}")
plt.legend()
plt.title("YAMNet Embeddings (PCA)")
plt.show()
