"""
    ディレクトリ内のすべてのwavファイルを取得してcsvファイルに変換する
    wavファイルの名前とlabelを対応させたcsvファイルを作成する
"""

import os
import pandas as pd
import re

# 入力ディレクトリの設定
input_dir = '../data'

# メタデータを格納するリスト
metadata = []

# ファイル名からラベルを抽出する関数
def extract_label(filename):
    label = re.sub(r'\(.*\)', '', filename)  # ()内の文字を除去
    label = os.path.splitext(label)[0]  # 拡張子を除去
    return label

# ディレクトリ内のファイルを走査
for filename in os.listdir(input_dir):
    if filename.endswith('.wav'):
        label = extract_label(filename)
        metadata.append({'filename': filename, 'label': label})

# メタデータをDataFrameに変換
df = pd.DataFrame(metadata)

# CSVファイルに書き出し
output_file = 'metadata.csv'
df.to_csv(output_file, index=False)

print(f"Metadata saved to {output_file}")