import os
import pandas as pd
import soundfile as sf
from sklearn.model_selection import train_test_split

# データディレクトリの設定
data_dir = '../data'
metadata_file = './metadata.csv'  # メタデータファイルにはファイル名とラベルが含まれる

# メタデータの読み込み
metadata = pd.read_csv(metadata_file)

# データセットの分割
train_metadata, test_metadata = train_test_split(metadata, test_size=0.2, random_state=42)

# データローダー用のデータセット作成
def create_dataset(metadata, data_dir):
    dataset = []
    for index, row in metadata.iterrows():
        file_path = os.path.join(data_dir, row['filename'])
        label = row['label']
        dataset.append({'audio': file_path, 'label': label})
    return dataset

train_dataset = create_dataset(train_metadata, data_dir)
test_dataset = create_dataset(test_metadata, data_dir)

# データセットの保存
import pickle
with open('train_dataset.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)
with open('test_dataset.pkl', 'wb') as f:
    pickle.dump(test_dataset, f)

print("データセットの準備が完了しました。")