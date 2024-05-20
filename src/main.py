import torch
from transformers import Wav2Vec2Processor, HubertModel, TrainingArguments, Trainer
from datasets import load_from_disk

# データセットの読み込み
dataset = load_from_disk("../dataset")

# プロセッサとモデルのロード
processor = Wav2Vec2Processor.from_pretrained("rinna/japanese-hubert-base")
model = HubertModel.from_pretrained("rinna/japanese-hubert-base")

# データローダの作成
def preprocess_function(examples):
    audio = sf.read(examples["audio"])[0]
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    return inputs

encoded_dataset = dataset.map(preprocess_function, remove_columns=["audio"])

# トレーニング設定
training_args = TrainingArguments(
    output_dir="../results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["test"],
)

# GPUが利用可能か確認
if torch.cuda.is_available():
    model.cuda()

trainer.train()