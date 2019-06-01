import json
import os

test_pth = 'test.json'
train_pth = 'train.json'

def load_all_data(json_path):
    all_data = []
    with open(json_path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        all_data.append(json.loads(line))
    return all_data

print(f"test set len: {len(load_all_data(test_pth))}")
print(f"train set len: {len(load_all_data(train_pth))}")