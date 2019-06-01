import pandas as pd

def create_features_and_split(all_data):
    featurized_data = create_features(all_data)
    split_train_val_data = split_train_val(featurized_dataframe)
    return split_train_val_data

def create_features(all_data):
    df = pd.DataFrame(data) #load into dataframe
    return df

def split_train_val_data(featurized_dataframe):
    # TODO pandas magic here
    return featurized_dataframe



