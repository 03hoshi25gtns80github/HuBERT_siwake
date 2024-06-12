"""
最後の層を削除し、途中の全結合層から埋め込みを取得しクラスタリングを行う
"""

import sys
import os

# pyclusteringディレクトリをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyclustering'))

# その後、通常通りにインポート
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer

import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, HubertModel
import torch
import pandas as pd
from torch import nn

# カスタム分類モデルの定義（FC3層を削除）
class ClassificationModel(nn.Module):
    def __init__(self, hubert_model):
        super(ClassificationModel, self).__init__()
        self.hubert = hubert_model
        self.fc1 = nn.Linear(self.hubert.config.hidden_size * 964, 512)  # 964はシーケンス長の例
        self.fc2 = nn.Linear(512, 256)

    def forward(self, input_values):
        if input_values.dim() == 4:
            input_values = input_values.squeeze(1)
        
        outputs = self.hubert(input_values)
        hidden_states = outputs.last_hidden_state
        
        # フラット化
        x = hidden_states.view(hidden_states.size(0), -1)
        
        x = self.fc1(x)
        x = torch.relu(x)  # 活性化関数としてReLUを使用
        
        x = self.fc2(x)
        x = torch.relu(x)  # 活性化関数としてReLUを使用
        
        return x  # FC2層の出力を返す

# モデルとプロセッサのロード
model_name = "rinna/japanese-hubert-base"
hubert_model = HubertModel.from_pretrained(model_name)
model = ClassificationModel(hubert_model)

# 保存されたモデルのパラメータをロード（CPUにマッピング）
state_dict = torch.load("finetuned_hubert_model.pth", map_location=torch.device('cpu'))

# FC3層のパラメータを除去
state_dict = {k: v for k, v in state_dict.items() if 'fc3' not in k}

# 新しいモデルに適用
model.load_state_dict(state_dict, strict=False)
model.eval()  # 評価モードに切り替え

# ディレクトリ内のすべてのwavファイルを取得##########################################
data_dir = "../test2"
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
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    inputs = feature_extractor(
        raw_speech_16kHz,
        return_tensors="pt",
        sampling_rate=16000,
    )
    
    # 入力データの形状を変換
    inputs['input_values'] = inputs['input_values'].transpose(0, 1)
    
    # モデルに入力してFC2層の出力を取得
    with torch.no_grad():
        fc2_output = model(inputs['input_values'])
    
    # 特徴量リストに追加
    features_list.append(fc2_output.squeeze().detach().numpy())
    
    # ファイルネームリストに追加
    filenames.extend([os.path.basename(wav_file)])

# K-meansクラスタリング
# クラスタ数の設定#########################################
num_clusters = 3

# 初期クラスタの設定
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

# データフレームをクラスタ順に並び替え
df = df.sort_values(by='cluster')

# メタデータを表示
print("Metadata DataFrame with Clustering Results:")
print(df)

# メタデータをCSVファイルとして保存
df.to_csv('metadata_with_clusters.csv', index=False)
print("Metadata with clustering results saved to metadata_with_clusters.csv")