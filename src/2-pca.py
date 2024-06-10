import sys
import os

# pyclusteringディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyclustering'))

# その後、通常通りにインポート
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer

import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, AutoModel
import torch
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

# モデルとプロセッサのロード
model_name = "rinna/japanese-hubert-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# ディレクトリ内のすべてのwavファイルを取得
data_dir = "../wav"
wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]

# 特徴量リスト
features_list = []

# ファイルネームリスト
filenames = []

for wav_file in wav_files:
    audio_file = os.path.join(data_dir, wav_file)
    
    # wavファイルの読み込み
    raw_speech, sr = sf.read(audio_file)
    
    # サンプリングレートの変換
    if sr != 16000:
        raw_speech_16kHz = librosa.resample(raw_speech, orig_sr=sr, target_sr=16000)
    else:
        raw_speech_16kHz = raw_speech
    
    # 入力データの長さを確認
    print(f"Input data length for {wav_file}: {len(raw_speech_16kHz)}")
    
    # 特徴量の抽出
    inputs = feature_extractor(
        raw_speech_16kHz,
        return_tensors="pt",
        sampling_rate=16000,
    )
    
    # 入力データの形状を変換
    inputs['input_values'] = inputs['input_values'].transpose(0, 1)
    outputs = model(**inputs)
    
    # 埋め込み表現の取得
    embeddings = outputs.last_hidden_state
    print(f"Output shape for {wav_file}: {embeddings.size()}")  # [1, #frames, 768]
    
    # 特徴量リストに追加
    features_list.append(embeddings.squeeze().detach().numpy())
    
    # ファイルネームリストに追加
    filenames.extend([os.path.basename(wav_file)])

# 特徴量リストを2次元配列に変換
features_array = np.vstack(features_list)

# PCAの実行
pca = PCA(n_components=1024)  # 2次元に削減
pca_features = pca.fit_transform(features_array)

# X-meansクラスタリング
# ハイパーパラメータの設定
min_clusters = 3
max_clusters = 4

# 初期クラスタの設定
initial_centers = kmeans_plusplus_initializer(pca_features, min_clusters).initialize()
xmeans_instance = xmeans(pca_features, initial_centers, max_clusters)
xmeans_instance.process()
clusters = xmeans_instance.get_clusters()
labels = [0] * len(pca_features)

for cluster_id, cluster in enumerate(clusters):
    for index in cluster:
        labels[index] = cluster_id

# クラスタリング結果をメタデータに追加
df = pd.DataFrame({'filename': filenames, 'cluster': labels})

# データフレームをクラスタ順に並び替え
df = df.sort_values(by='cluster')

# メタデータを表示
print("Metadata DataFrame with Clustering Results:")
print(df)

# メタデータをCSVファイルとして保存
df.to_csv('metadata_with_clusters.csv', index=False)
print("Metadata with clustering results saved to metadata_with_clusters.csv")