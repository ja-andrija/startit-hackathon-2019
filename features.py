import pandas as pd
import util
import numpy as np
import ast

def create_features_and_split(all_data):
    featurized_dataframe = create_features(all_data)
    split_dataframe = split_train_val_data(featurized_dataframe)
    return split_dataframe

def mdns_to_token_appereance(mdns_list):
    mdns_tokens = util.get_mdns_tokens()
    appereance = np.zeros(len(mdns_tokens), dtype=int)
    if mdns_list == 'nan':
        return appereance
    mdns_list = ast.literal_eval(mdns_list)
    for mdns in mdns_list:
        if mdns in mdns_tokens:
            appereance[mdns_tokens[mdns]] = 1
    return appereance.tolist()

def create_features(all_data):
    df = pd.DataFrame(all_data) #load into dataframe
    
    df['has_upnp'] = df['upnp'].notna()
    df['has_ssdp'] = df['ssdp'].notna()
    df['has_mdns'] = df['mdns_services'].notna()
    df['mac_first_3_bytes'] = [mac[0:8] for mac in df['mac']]

    df['mdns_tokens'] = [mdns_to_token_appereance(str(mdns_list)) for mdns_list in df['mdns_services']]

    return df[['mac_first_3_bytes', 'has_upnp', 'has_ssdp', 'has_mdns', 'device_class', 'mdns_tokens']]

def split_train_val_data(featurized_dataframe):
    # TODO pandas magic here
    return featurized_dataframe


# sanity checks
df = util.load_data_to_dataframe('dataset/train.json')
splitdf = create_features_and_split(df)
print(splitdf.head(20))
print(splitdf.groupby('device_class').sum())
