import pandas as pd
import json
import matplotlib.pyplot as plt
from pyitlib import discrete_random_variable as drv
import numpy as np
import math
from dataset_preparation.util import CLASSES_TO_INT
from tqdm import tqdm

def extract_port(x):
    if type(x) is dict:
        return int(x['port'])
    return None

def contains_port(x, port):
    if type(x) is list:
        if {'port': port, 'protocol':'udp'} in x or {'port': port, 'protocol':'tcp'} in x:
            return True
    return False 

def calculate_gain(port, df, h_x, pbar):
    X = df['device_class'].apply(lambda x: CLASSES_TO_INT[x]).values.astype(int)
    Y = df['services'].apply(lambda x: contains_port(x, port)).values.astype(int)
    h_x_y = drv.entropy_conditional(X, Y)
    i_x_y = h_x - h_x_y
    pbar.update(1)
    return i_x_y

# read the entire file into a python array
def compute_port_statistics():
    with open("D:\\dataset\\train.json", 'rb') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data] #convert string to dict format
    df = pd.DataFrame(data) #load into dataframe

    print("> Flattening ports and computing frequencies")
    services_df = df.services.apply(pd.Series)
    flattened_services = services_df.merge(df, left_index=True, right_index=True).drop(['services'], axis=1).melt(id_vars=['device_class', 'device_id'], value_name="services").drop(['variable'], axis=1).dropna()
    flattened_services['services'] = flattened_services.apply(lambda x: extract_port(x['services']), axis=1)
    flattened_services = flattened_services.dropna()

    h_x = drv.entropy(df['device_class'])

    port_stats = flattened_services.groupby(['services'])['device_id'].agg(
        {"count": len}).sort_values(
        "count", ascending=False).reset_index()

    print("> Computing Information Gain")
    with tqdm(total=len(port_stats.index)) as pbar:
        port_stats['gain'] = port_stats.apply(lambda x: calculate_gain(x['services'], df, h_x, pbar), axis=1)

    port_stats.to_csv('cache/port_counts.csv', index=False)



