import pandas as pd
import util

def create_features_and_split(all_data):
    featurized_dataframe = create_features(all_data)
    split_dataframe = split_train_val_data(featurized_dataframe)
    return split_dataframe

def create_features(all_data):
    df = pd.DataFrame(all_data) #load into dataframe
    
    df['has_upnp'] = df['upnp'].notna()
    df['has_ssdp'] = df['ssdp'].notna()
    df['has_mdns'] = df['mdns_services'].notna()
    df['mac_first_3_bytes'] = [mac[0:8] for mac in df['mac']]

    return df[['mac_first_3_bytes', 'has_upnp', 'has_ssdp', 'has_mdns', 'device_class']]

def split_train_val_data(featurized_dataframe):
    # TODO pandas magic here
    return featurized_dataframe


# sanity checks
df = util.load_data_to_dataframe('dataset/train.json')
splitdf = create_features_and_split(df)
print(splitdf.head())
print(splitdf.groupby('device_class').sum())
