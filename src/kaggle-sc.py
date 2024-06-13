"""
jpynbで実行する
教師ありでファインチューニング
SequenceClassificationモデルで分類も代を解く
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, HubertForSequenceClassification
import pickle
import soundfile as sf
import librosa
from sklearn.metrics import accuracy_score

# データセットのロード
with open('train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

with open('test_dataset.pkl', 'rb') as f:
    test_dataset = pickle.load(f)

# ラベルを動的に取得してマッピングする辞書を作成
unique_labels = set(item['label'] for item in train_dataset)
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}

# カスタムデータセットクラスの定義
class AudioDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("rinna/japanese-hubert-base")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        audio_path = self.dataset[idx]['audio']
        label = self.dataset[idx]['label']
        raw_speech, sr = sf.read(audio_path)
        if sr != 16000:
            raw_speech = librosa.resample(raw_speech, orig_sr=sr, target_sr=16000)
        inputs = self.feature_extractor(raw_speech, return_tensors="pt", sampling_rate=16000)
        inputs['input_values'] = inputs['input_values'].squeeze(1)
        return inputs['input_values'], label_to_int[label]

# データローダーの設定
train_loader = DataLoader(AudioDataset(train_dataset), batch_size=1, shuffle=True)
test_loader = DataLoader(AudioDataset(test_dataset), batch_size=1, shuffle=False)

# モデルのロード
model_name = "rinna/japanese-hubert-base"
model = HubertForSequenceClassification.from_pretrained(model_name, num_labels=len(unique_labels))

# 損失関数と最適化手法の設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# ファインチューニングの実行
num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# ファインチューニング後のモデルを保存
torch.save(model.state_dict(), "finetuned_hubert_model.pth")
print("ファインチューニングが完了しました。")

# テストデータで精度を確認
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"テストデータでの精度: {accuracy * 100:.2f}%")