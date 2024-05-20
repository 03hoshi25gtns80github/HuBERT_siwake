import os
import random
import soundfile as sf
from datasets import Dataset, DatasetDict

def create_dataset(data_dir, split_ratio=0.8):
    data_files = {"train": [], "test": []}
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".wav")]
    random.shuffle(all_files)
    
    split_index = int(len(all_files) * split_ratio)
    train_files = all_files[:split_index]
    test_files = all_files[split_index:]
    
    for file_name in train_files:
        file_path = os.path.join(data_dir, file_name)
        data_files["train"].append({"audio": file_path})
    
    for file_name in test_files:
        file_path = os.path.join(data_dir, file_name)
        data_files["test"].append({"audio": file_path})
    
    dataset = DatasetDict({
        "train": Dataset.from_list(data_files["train"]),
        "test": Dataset.from_list(data_files["test"])
    })
    
    return dataset

if __name__ == "__main__":
    data_dir = "../data"
    dataset = create_dataset(data_dir)
    dataset.save_to_disk("../data/dataset")