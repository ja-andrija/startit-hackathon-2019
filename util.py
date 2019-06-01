import json
import os
import pandas
import random

test_pth = 'test.json'
train_pth = 'train.json'

def load_all_data(json_path):
    all_data = []
    with open(json_path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        all_data.append(json.loads(line))
    return all_data

def load_data_to_dataframe(json_path):
    # read the entire file into a python array
    with open(json_path, 'rb') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data] #convert string to dict format
        return pandas.DataFrame(data) #load into dataframe

def write_json_to_file(j_object, file_path):
    with open(file_path, 'w') as fp:
        json.dump(j_object, fp)

def read_proper_json_from_file(json_path):
    with open(json_path, 'r') as fp:
        return json.load(fp)

def split_train_val(json_path, train_ratio = 0.8):
    data = load_all_data(json_path)
    random.seed(42)
    random.shuffle(data)
    train_split_len = int(len(data) * train_ratio)
    train_split = data[:train_split_len]
    val_split = data[train_split_len:]
    write_json_to_file(train_split, 'train_split.json')
    write_json_to_file(val_split, 'val_split.json')