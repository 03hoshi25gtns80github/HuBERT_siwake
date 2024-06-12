"""
jpynbで実行する
教師ありでファインチューニング
state_dictを保存する
"""

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, HubertModel
import pickle
import soundfile as sf
import librosa

# データセットのロード
with open('train_dataset.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

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
        inputs['input_values'] = inputs['input_values'].transpose(0, 1)
        return inputs['input_values'], label_to_int[label]  # ラベルを整数に変換して返す

# データローダーの設定
train_loader = DataLoader(AudioDataset(train_dataset), batch_size=1, shuffle=True)

# モデルとプロセッサのロード
model_name = "rinna/japanese-hubert-base"
hubert_model = HubertModel.from_pretrained(model_name)

# カスタム分類モデルの定義
class ClassificationModel(nn.Module):
    def __init__(self, hubert_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.hubert = hubert_model
        self.fc1 = nn.Linear(self.hubert.config.hidden_size * 964, 512)  # 964はシーケンス長の例
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_labels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_values):
        print(f"Input shape before squeeze: {input_values.shape}")
        if input_values.dim() == 4:
            input_values = input_values.squeeze(1)
        print(f"Input shape after squeeze: {input_values.shape}")
        
        outputs = self.hubert(input_values)
        hidden_states = outputs.last_hidden_state
        print(f"Hidden states shape: {hidden_states.shape}")
        
        # フラット化
        x = hidden_states.view(hidden_states.size(0), -1)
        print(f"Shape after flattening: {x.shape}")
        
        x = self.fc1(x)
        print(f"Shape after FC1 layer: {x.shape}")
        
        x = torch.relu(x)  # 活性化関数としてReLUを使用
        x = self.fc2(x)
        print(f"Shape after FC2 layer: {x.shape}")
        
        x = torch.relu(x)  # 活性化関数としてReLUを使用
        logits = self.fc3(x)
        print(f"Shape after FC3 layer: {logits.shape}")
        
        probs = self.softmax(logits)
        print(f"Output shape: {probs.shape}")
        
        return probs

num_labels = len(unique_labels)  # クラス数を動的に設定
model = ClassificationModel(hubert_model, num_labels)

# 損失関数と最適化手法の設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# ファインチューニングの実行
num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # デバッグ用のprint文を追加
        print(f"Batch input shape before squeeze: {inputs.shape}")
        inputs = inputs.squeeze(1)  # 形状を (batch_size, sequence_length) に整形
        print(f"Batch input shape after squeeze: {inputs.shape}")
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
         # labelsをテンソルに変換し、形状を (batch_size,) に整形
        if isinstance(labels, tuple):
            labels = torch.tensor(labels)
        labels = labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# ファインチューニング後のモデルを保存
torch.save(model.state_dict(), "finetuned_hubert_model.pth")
print("ファインチューニングが完了しました。")