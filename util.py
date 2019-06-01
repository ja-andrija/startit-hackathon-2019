import json
import os
import pandas
import random
from collections import defaultdict
from sklearn.utils import resample
import sys

test_pth = 'test.json'
train_pth = 'train.json'

device_names = [
    'AUDIO',
    'GAME_CONSOLE',
    'HOME_AUTOMATION',
    'IP_PHONE',
    'MEDIA_BOX',
    'MOBILE',
    'NAS',
    'PC',
    'PRINTER',
    'SURVEILLANCE',
    'TV',
    'VOICE_ASSISTANT',
    'GENERIC_IOT']

def load_all_data(json_path):
    all_data = []
    with open(json_path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        all_data.append(json.loads(line))
    return all_data

def load_data_to_dataframe(json_path, upsample = False):
    # read the entire file into a python array
    with open(json_path, 'rb') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data] #convert string to dict format
        if upsample:
            balance_dataset(data)
        return pandas.DataFrame(data) #load into dataframe

def balance_dataset(data, upsampling = True):
    separated_classes_data = separate_data_into_classes(data)

    max_count = 0
    min_count = -10000000
    random.seed(42)

    for device_class, device_data in separated_classes_data.items():
        current_data_len = len(device_data)

        if max_count < current_data_len:
            max_count = current_data_len

        if min_count > current_data_len:
            min_count = current_data_len

    if upsampling:
        for device_class, device_data in separated_classes_data.items():
            new_data = list()

            while len(device_data) + len(new_data) < max_count:
                rand_num = random.randint(0, len(device_data) - 1)
                new_data.append(device_data[rand_num])

            device_data.extend(new_data)
            separated_classes_data[device_class] = device_data
    
    data_balanced = list()

    for device_class, device_data in separated_classes_data.items():
        data_balanced.extend(device_data)

    return random.shuffle(data_balanced)


def separate_data_into_classes(data):
    separated_data = dict()
    for device_name in device_names:
        separated_data[device_name] = list()
    
    for dp in data:
        separated_data[dp["device_class"]].append(dp)

    return separated_data

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
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_out", "mdns_tokens.json")) as f:
        mdns_tokens = json.load(f)
    return mdns_tokens

def enumerate_dhcp_classid():
    class_ids = set()

    stat_tag_device = read_proper_json_from_file(r'C:\Users\vr1\startit-hackathon-2019\key_stats_dhcp_tags_dev.json')
    i = 0
    
    for class_id, val in stat_tag_device["classid"].items():
        class_ids.add(class_id)

    tokenized = dict(list(enumerate(class_ids)))
    inv = {v: k for k, v in tokenized.items()}
    with open(r'C:\Users\vr1\startit-hackathon-2019\tockenized.json', 'w') as f:
        json.dump(inv, f)

def get_dhcp_classid_tokens():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "stats_out", r'C:\Users\vr1\startit-hackathon-2019\tockenized.json')) as f:
        dhcp_classid_tokens = json.load(f)
    return dhcp_classid_tokens
