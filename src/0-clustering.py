import sys
import os

# pyclusteringディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyclustering'))

# その後、通常通りにインポート
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import soundfile as sf
import librosa
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModelForSequenceClassification
import pickle


# データセットのロード
with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

# カスタムデータセットクラスの定義
class AudioDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("rinna/japanese-hubert-base")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_path = self.dataset[idx]['audio']
        raw_speech, sr = sf.read(audio_path)
        if sr != 16000:
            raw_speech = librosa.resample(raw_speech, orig_sr=sr, target_sr=16000)
        inputs = self.feature_extractor(raw_speech, return_tensors="pt", sampling_rate=16000)
        return inputs['input_values'].squeeze(0), self.dataset[idx]['audio']

# データローダーの設定
test_loader = DataLoader(AudioDataset(test_dataset), batch_size=1, shuffle=False)

# ファインチューニングしたモデルのロード
model_name = "rinna/japanese-hubert-base"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=10)
model.load_state_dict(torch.load("finetuned_hubert_model.pth"))
model.eval()

# 特徴量抽出用のモデル設定
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.model.classifier = nn.Identity()  # 全結合層を無効化

    def forward(self, input_values):
        outputs = self.model(input_values)
        return outputs.last_hidden_state

feature_extractor_model = FeatureExtractor(model)
feature_extractor_model.eval()

# 特徴量抽出
features_list = []
filenames = []
for inputs, filename in test_loader:
    with torch.no_grad():
        features = feature_extractor_model(inputs)
        pooled_features = torch.mean(features, dim=1)
        features_list.append(pooled_features.squeeze().detach().numpy())
        filenames.append(filename[0])

# K-meansクラスタリング
num_clusters = 4  # 例として4クラスタ
initial_centers = kmeans_plusplus_initializer(features_list, num_clusters).initialize()
kmeans_instance = kmeans(features_list, initial_centers)
kmeans_instance.process()
clusters = kmeans_instance.get_clusters()
labels = [0] * len(features_list)

for cluster_id, cluster in enumerate(clusters):
    for index in cluster:
        labels[index] = cluster_id

# クラスタリング結果をメタデータに追加
df = pd.DataFrame({'filename': filenames, 'cluster': labels})
df = df.sort_values(by='cluster')
df.to_csv('clustering_results_metadata.csv', index=False)
print("Clustering results metadata saved to clustering_results_metadata.csv")
