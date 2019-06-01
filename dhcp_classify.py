from util import read_proper_json_from_file, write_json_to_file
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input

CLASSES_TO_INT = {
    'AUDIO': 0,
    'GAME_CONSOLE': 1,
    'HOME_AUTOMATION': 2,
    'IP_PHONE':3,
    'MEDIA_BOX':4,
    'MOBILE':5,
    'NAS':6,
    'PC':7,
    'PRINTER':8,
    'SURVEILLANCE':9,
    'TV':10,
    'VOICE_ASSISTANT':11,
    'GENERIC_IOT':12
}

ONEHOT_VECTOR_SIZE = 256

def extract_only_dhcp(json_path):
    data = read_proper_json_from_file(json_path)
    dhcp_data = []
    for dp in data:
        if 'dhcp' in dp.keys():
            dhcp_data.append(dp)
    filename = os.path.basename(json_path)
    filename = os.path.splitext(filename)[0]
    write_json_to_file(dhcp_data, f"{filename}_dhcp_only.json")
    return dhcp_data

def get_one_hot_encoding(json_path):
    data = extract_only_dhcp(json_path)
    input_data = []
    input_class = []
    for dp in data:
        if 'paramlist' in dp['dhcp'][0]:
            one_hot = ONEHOT_VECTOR_SIZE * [0]
            one_hot_labels = len(CLASSES_TO_INT) * [0]
            params = [int(v) for v in dp['dhcp'][0]['paramlist'].split(',')]
            assert max(params) <= ONEHOT_VECTOR_SIZE
            for e in params:
                one_hot[e] = 1
            one_hot_labels[CLASSES_TO_INT[dp['device_class']]] = 1
            input_data.append(one_hot)
            input_class.append(one_hot_labels)
    return input_data, input_class


def create_model():
    a = Sequential()
    a.add(Dense(256, activation = 'tanh'))
    a.add(Dense(256, activation = 'tanh'))
    a.add(Dense(256, activation = 'tanh'))
    a.add(Dense(256, activation = 'tanh'))
    a.add(Dense(256, activation = 'tanh'))
    a.add(Dense(len(CLASSES_TO_INT), activation='softmax'))
    return a

def train():
    train_input, train_labels = get_one_hot_encoding('train_split.json')
    train_input = np.array(train_input)
    train_labels = np.array(train_labels)
    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_input, y=train_labels, epochs= 10000)

if __name__ == "__main__":
    train()