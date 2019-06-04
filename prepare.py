import sys
import argparse
import os
import random
import shutil

def split_train_test(json_path, train_ratio):
    with open(json_path, 'r') as fp:
        data = fp.readlines()
    random.seed()
    random.shuffle(data)
    train_split_len = int(len(data) * train_ratio)
    train_split = data[:train_split_len]
    test_split = data[train_split_len:]
    with open('cache/train_split.json', 'w') as f:
        f.writelines(train_split)
    with open('dataset/test_split.json', 'w') as f:
        f.writelines(test_split)

if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument('--dataset', '-d', help='Dataset folder path with train.json and test.json files', type=str)
    parser.add_argument('--train_ratio', '-r', help='Train to test ratio used for split', type=float, default=0.8)
    
    args = parser.parse_args()

    dataset_path = args.dataset
    split_ratio = args.train_ratio

    # Copy test.json to cache/test.json
    shutil.copy2(os.path.join(dataset_path, "test.json"), "/cache/test.json")

    # Split train.json into train_split.json and test_split.json
    split_train_test(os.path.join(dataset_path, "train.json"), split_ratio)