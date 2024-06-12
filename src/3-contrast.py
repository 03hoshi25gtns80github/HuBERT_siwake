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
        inputs['input_values'] = inputs['input_values'].transpose(0, 1)
        return inputs['input_values']

# データローダーの設定
train_loader = DataLoader(AudioDataset(train_dataset), batch_size=1, shuffle=True)

# モデルとプロセッサのロード
model_name = "rinna/japanese-hubert-base"
hubert_model = HubertModel.from_pretrained(model_name)

# カスタムコントラスト学習モデルの定義
class ContrastiveModel(nn.Module):
    def __init__(self, hubert_model):
        super(ContrastiveModel, self).__init__()
        self.hubert = hubert_model
        self.fc1 = nn.Linear(self.hubert.config.hidden_size * 964, 512)  # 964はシーケンス長の例

    def forward(self, input_values):
        if input_values.dim() == 4:
            input_values = input_values.squeeze(1)
        
        outputs = self.hubert(input_values)
        hidden_states = outputs.last_hidden_state
        
        # フラット化
        x = hidden_states.view(hidden_states.size(0), -1)
        
        x = self.fc1(x)
        
        return x

model = ContrastiveModel(hubert_model)

# 損失関数と最適化手法の設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# コントラスト学習の実行
num_epochs = 1
model.train()
for epoch in range(num_epochs):
    for inputs in train_loader:
        inputs = inputs.squeeze(1)  # 形状を (batch_size, sequence_length) に整形
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # コントラスト学習の損失計算（例としてクロスエントロピーを使用）
        # 実際のコントラスト学習では、適切な損失関数を使用する必要があります
        labels = torch.arange(outputs.size(0))  # ダミーラベル
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# 学習後のモデルを保存
torch.save(model.state_dict(), "contrastive_hubert_model.pth")
print("コントラスト学習が完了しました。")