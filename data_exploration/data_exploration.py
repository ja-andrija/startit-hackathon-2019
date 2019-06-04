import pandas as pd
import json
import matplotlib.pyplot as plt

def plot_class_distribution(df):
    df['device_class'].value_counts().plot(kind='bar')
    plt.show()

# read the entire file into a python array
with open('dataset/train.json', 'rb') as f:
    data = f.readlines()
    data = [json.loads(line) for line in data] #convert string to dict format
    df = pd.DataFrame(data) #load into dataframe


    df['has_upnp'] = df['upnp'].notna()
    df['has_ssdp'] = df['ssdp'].notna()

    # df.drop(['upnp', 'ssdp'], 1).to_csv('excel rules 2.csv')

    print(df.groupby('device_class').aggregate("count"))
    grb = df.groupby(['device_class', 'has_upnp']).aggregate('count')

    # grb.to_csv('grb.csv')

    print(df.head())