import pandas as pd
import util
import numpy as np
import ast

def create_features_and_split(all_data):
    featurized_dataframe = create_features(all_data)
    split_dataframe = split_train_val_data(featurized_dataframe)
    return split_dataframe

def mdns_to_token_appereance(mdns_list, mdns_tokens):
    appereance = np.zeros(len(mdns_tokens), dtype=int)
    if mdns_list == 'nan':
        return appereance
    mdns_list = ast.literal_eval(mdns_list)
    for mdns in mdns_list:
        if mdns in mdns_tokens:
            appereance[mdns_tokens[mdns]] = 1
    return appereance.tolist()

def add_mdns_tokens_get_token_names_list(df):
    mdns_tokens = util.get_mdns_tokens()
    mdns_tokens_df = [mdns_to_token_appereance(str(mdns_list), mdns_tokens) for mdns_list in df['mdns_services']]
    mdns_token_names_list = []
    for i in range(len(mdns_tokens)):
        df[f'mdns_token_{i}'] = [mdns_tokens_list[i] for mdns_tokens_list in mdns_tokens_df]
        mdns_token_names_list.append(f'mdns_token_{i}')
    return mdns_token_names_list

def create_features(all_data):
    df = pd.DataFrame(all_data) #load into dataframe
    
    df['has_upnp'] = df['upnp'].notna()
    df['has_ssdp'] = df['ssdp'].notna()
    df['has_mdns'] = df['mdns_services'].notna()
    df['mac_first_3_bytes'] = [mac[0:8] for mac in df['mac']]

    mdns_token_names_list = add_mdns_tokens_get_token_names_list(df)
    
    return df[['mac_first_3_bytes', 'has_upnp', 'has_ssdp', 'has_mdns', 'device_class'] + mdns_token_names_list]

def split_train_val_data(featurized_dataframe):
    # TODO pandas magic here
    return featurized_dataframe


# sanity checks
df = util.load_data_to_dataframe('dataset/train.json')
splitdf = create_features_and_split(df)
print(splitdf.head(20))
print(splitdf.groupby('device_class').sum())
