import pandas as pd
import json
import matplotlib.pyplot as plt
from pyitlib import discrete_random_variable as drv
import numpy as np
import math

def plot_class_distribution(df):
    df['device_class'].value_counts().plot(kind='bar')
    plt.show()

def extract_port(x):
    if type(x) is dict:
        return int(x['port'])
    return None

def calculate_gain(port, flattened_services, h_x):
    n = len(flattened_services.index)
    port_contains = flattened_services
    port_contains['has_port'] = flattened_services['services']==port
    port_df = port_contains.loc[port_contains['has_port']==True]
    no_port_df = port_contains.loc[port_contains['has_port']==False]

    p_x1 = len(port_df.index) / n
    p_x0 = 1 - p_x1

    # Compute h_y_x0
    p_y_x0 = no_port_df.groupby('device_class').agg('count')
    p_y_x0['p'] = p_y_x0.apply(lambda x: x['services'] / n, axis=1)
    p_y_x0['q'] = p_y_x0.apply(lambda x: -x['p']*math.log2(x['p']), axis=1)
    h_y_x0 = p_x0 * p_y_x0['q'].aggregate('sum')

    # Compute h_y_x1
    p_y_x1 = port_df.groupby('device_class').agg('count')
    p_y_x1['p'] = p_y_x1.apply(lambda x: x['services'] / n, axis=1)
    p_y_x1['q'] = p_y_x1.apply(lambda x: -x['p']*math.log2(x['p']), axis=1)
    h_y_x1 = p_x1 * p_y_x1['q'].aggregate('sum')

    # Compute information gain
    h_y_x = h_y_x0 + h_y_x1
    i_x_y = h_x - h_y_x
    calculate_gain.counter += 1

    if calculate_gain.counter % 50 == 0:
        print(calculate_gain.counter)
    return i_x_y
calculate_gain.counter = 0

# read the entire file into a python array
def compute_gains():
    with open('train.json', 'rb') as f:
        data = f.readlines()
        data = [json.loads(line) for line in data] #convert string to dict format
        df = pd.DataFrame(data) #load into dataframe

        services_df = df.services.apply(pd.Series)
        flattened_services = services_df.merge(df, left_index=True, right_index=True).drop(['services'], axis=1).melt(id_vars=['device_class', 'device_id'], value_name="services").drop(['variable'], axis=1).dropna()
        flattened_services['services'] = flattened_services.apply(lambda x: extract_port(x['services']), axis=1)
        flattened_services = flattened_services.dropna()

        h_x = drv.entropy(df['device_class'])

        information_gain = []
        services_top_freq = flattened_services.groupby(['services'])['device_id'].agg(
            {"service_count": len}).sort_values(
            "service_count", ascending=False).head(4000).reset_index()
        #services = flattened_services.groupby('services').agg('count').nlargest(500, columns=['device_id']) 

        services_top_freq.apply(lambda x: information_gain.append(tuple((x['services'], calculate_gain(x['services'], flattened_services, h_x)))), axis=1)

        information_gain.sort(key=lambda p: p[1], reverse=True)
        with open('port_gains.json', 'w') as f:
            print(information_gain)
            json.dump(information_gain, f) # '[1, 2, [3, 4]]'

with open('train.json', 'rb') as f:
    data = f.readlines()
    data = [json.loads(line) for line in data] #convert string to dict format
    df = pd.DataFrame(data) #load into dataframe

    services_df = df.services.apply(pd.Series)
    flattened_services = services_df.merge(df, left_index=True, right_index=True).drop(['services'], axis=1).melt(id_vars=['device_class', 'device_id'], value_name="services").drop(['variable'], axis=1).dropna()
    flattened_services['services'] = flattened_services.apply(lambda x: extract_port(x['services']), axis=1)
    flattened_services = flattened_services.dropna()

    services_top_freq = flattened_services.groupby(['services'])['device_id'].agg(
        {"service_count": len}).sort_values(
        "service_count", ascending=False).head(4000).reset_index()
