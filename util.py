import json
import os
import pandas
import random
from collections import defaultdict

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

def split_train_val_without_changes(json_path, train_ratio = 0.8):
    with open(json_path, 'r') as fp:
        data = fp.readlines()
    random.seed(42)
    random.shuffle(data)
    train_split_len = int(len(data) * train_ratio)
    train_split = data[:train_split_len]
    val_split = data[train_split_len:]
    with open('train_split_orig.json', 'w') as f:
        f.writelines(train_split)
    with open('val_split_orig.json', 'w') as f:
        f.writelines(val_split)

def make_tokens(in_json, out_json):
    with open(in_json, 'r') as f:
        data = json.load(f)
    tokenized = dict(list(enumerate(data)))
    inv = {v: k for k, v in tokenized.items()}
    with open(out_json, 'w') as f:
        json.dump(inv, f)

def get_mdns_tokens():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "mdns_tokens.json")) as f:
        mdns_tokens = json.load(f)
    return mdns_tokens

def get_mac_set(json_path):
    mac_set = defaultdict(int)
    with open(json_path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        mac_set[json.loads(line)['mac'][:8]] += 1
    return mac_set

def create_mac_tokens():
    train_mac_set = get_mac_set('dataset/train.json')
    test_mac_set = get_mac_set('dataset/test.json')
    intersect = set(train_mac_set.keys()) & set(test_mac_set.keys())
    tokenized = dict(list(enumerate(intersect)))
    inv = {v: k for k, v in tokenized.items()}
    with open('dataset/mac_tokens.json', 'w') as f:
        json.dump(inv, f)

def get_mac_tokens():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "mac_tokens.json")) as f:
        mac_tokens = json.load(f)
    return mac_tokens
