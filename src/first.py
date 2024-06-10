import soundfile as sf
import librosa
from transformers import AutoFeatureExtractor, AutoModel
import torch

# モデルとプロセッサのロード
model_name = "rinna/japanese-hubert-base"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

# wavファイルの読み込み
audio_file = "../data/yuriko2.wav"
raw_speech, sr = sf.read(audio_file)

# サンプリングレートの変換
if sr != 16000:
    raw_speech_16kHz = librosa.resample(raw_speech, orig_sr=sr, target_sr=16000)
else:
    raw_speech_16kHz = raw_speech
    
# 入力データの長さを確認
print(f"Input data length: {len(raw_speech_16kHz)}")

# 特徴量の抽出
inputs = feature_extractor(
    raw_speech_16kHz,
    return_tensors="pt",
    sampling_rate=16000,
)

# 入力データの形状を確認
print(f"Input values shape: {inputs['input_values'].shape}")

# 入力データの形状を変換
inputs['input_values'] = inputs['input_values'].transpose(0, 1)

# 変換後の入力データの形状を確認
print(f"Transformed input values shape: {inputs['input_values'].shape}")

outputs = model(**inputs)

# 埋め込み表現の取得
embeddings = outputs.last_hidden_state
print(f"Input shape:  {inputs.input_values.size()}")  # [1, #samples]
print(f"Output shape: {embeddings.size()}")  # [1, #frames, 768]

# 平均プーリングの実装
pooled_features = torch.mean(embeddings, dim=1)  # [1, 768]
print(f"Pooled features shape: {pooled_features.size()}")  # [1, 768]