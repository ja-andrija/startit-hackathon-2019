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

    print(df.groupby('device_class').aggregate("count"))
