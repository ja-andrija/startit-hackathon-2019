import csv
import pandas as pd
import util
import numpy as np
import ast
import words
import json

def contains_port(x, port):
    if type(x['services']) is list:
        if {'port': port, 'protocol':'udp'} in x['services'] or {'port': port, 'protocol':'tcp'} in x['services']:
            return True
    return False 

def count_ports(x):
    if type(x['services']) is list:
        return len(x['services'])
    return 0

def create_features_and_split(all_data):
    featurized_dataframe = create_features(all_data)
    train_df, val_df = split_train_val_data(featurized_dataframe)
    return train_df, val_df

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

def mac_to_token_appereance(mac, mac_tokens):
    appereance = np.zeros(len(mac_tokens), dtype=int)
    mac_first_3_bytes = mac[:8]
    if mac_first_3_bytes in mac_tokens:
        appereance[mac_tokens[mac_first_3_bytes]] = 1
    return appereance.tolist()

def add_mac_tokens_get_token_names_list(df):
    mac_tokens = util.get_mac_tokens()
    mac_tokens_df = [mac_to_token_appereance(mac, mac_tokens) for mac in df['mac']]
    mac_token_names_list = []
    for i in range(len(mac_tokens)):
        df[f'mac_token_{i}'] = [mac_tokens_list[i] for mac_tokens_list in mac_tokens_df]
        mac_token_names_list.append(f'mac_token_{i}')
    return mac_token_names_list

def mac_to_int(mac):
    mac = mac.split(":")
    mac = ''.join(mac[:3])
    return int(mac, 16)

def add_port_list(df):
    ports_to_add = ['5353', '80', '1900', '443', '445', '8080', '139', '515', '427', '9']
    for port in ports_to_add:
        df[port] = df.apply(lambda x: contains_port(x, int(port)), axis=1)
    return ports_to_add

def create_features(all_data):
    df = pd.DataFrame(all_data) #load into dataframe
    
    df['has_upnp'] = df['upnp'].notna()
    df['has_ssdp'] = df['ssdp'].notna()
    df['has_mdns'] = df['mdns_services'].notna()
    df['has_dhcp'] = df['dhcp'].notna()
    df['mac_first_3_bytes'] = [mac_to_int(mac) for mac in df['mac']]
    df['open_port_count'] = df.apply(lambda x: count_ports(x), axis=1)

    port_list = add_port_list(df)

    mdns_token_names_list = add_mdns_tokens_get_token_names_list(df)
    mac_token_names_list = add_mac_tokens_get_token_names_list(df)
    upnp_words_list = words.create_upnp_word_columns(df)

    dhcp_names = add_dhcp(df)
    ssdp_words_list = words.create_ssdp_word_columns(df)
    device_class_column = ['device_class'] if 'device_class' in df else []
    return df[['has_upnp', 'has_ssdp', 'has_mdns', 'has_dhcp', 'device_id', 'open_port_count'] + mdns_token_names_list + upnp_words_list + ssdp_words_list + device_class_column + dhcp_names + port_list + mac_token_names_list]

'''    selected_columns = np.array(pd.read_csv('selected_columns.csv').values).flatten().tolist()
    if 'device_class' not in df:
        selected_columns.remove('device_class')
    return all_features[selected_columns]'''
    #feature_scores = pd.read_csv('feature_importance.csv')
    #top_features=feature_scores.nlargest(378,'Score')
    #top_features_list = top_features['Specs'].values.tolist()
    #return all_features[top_features_list + device_class_column + ['device_id']]

def split_train_val_data(featurized_dataframe):
    # TODO pandas magic here
    msk = np.random.rand(len(featurized_dataframe)) < 0.8
    train = featurized_dataframe[msk]
    val = featurized_dataframe[~msk]
    return train, val

def get_dhcp_onehot(df):
    all_dhcp_onehot = []
    for dhcp in df['dhcp']:
        one_hot = 256 * [0]
        if str(dhcp) == 'nan':
            all_dhcp_onehot.append(one_hot)
        else:
            try:
                dhcp_data = json.loads(str(dhcp).replace("'",'"'))[0]
            except Exception:
                print(str(dhcp))
                dhcp_data = {}
            if 'paramlist' not in dhcp_data:
                all_dhcp_onehot.append(one_hot)
            else:
                paramlist = dhcp_data['paramlist']
                paramlist = [int(p) for p in paramlist.split(',')]
                #print(paramlist)
                for param in paramlist:
                    one_hot[param] = 1
                all_dhcp_onehot.append(one_hot)
    return all_dhcp_onehot

def add_dhcp(df):
    all_dhcp_onehot = get_dhcp_onehot(df)
    dhcp_names = []
    for i in range(256):
        feature = [dhcp_onehot[i] for dhcp_onehot in all_dhcp_onehot]
        #print(len(feature))
        df[f"dhcp_feat{i}"] = feature
        dhcp_names.append(f"dhcp_feat{i}")
    return dhcp_names

# sanity checks
# df = util.load_data_to_dataframe('dataset/train_split_orig.json')
# print(len(df))
# add_dhcp(df)
# for dhcp in df['dhcp']:
#     if str(dhcp) != 'nan':
#         print(str(dhcp))
#         dhcp_data = json.loads(str(dhcp).replace("'",'"'))[0]
#         if 'paramlist' in dhcp_data:
#             print(dhcp_data['paramlist'])
# train, val = create_features_and_split(df)
# print(train.head(20))
# print(f"TRAIN: {len(train)}")
# print(train.groupby('device_class').sum())
# print(f"VAL: {len(val)}")
# print(val.groupby('device_class').sum())