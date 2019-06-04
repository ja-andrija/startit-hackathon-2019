import util
import features
import pandas as pd
from features import create_features_and_split, create_features
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from keras_model import KerasModel

def filterout_mac_features(df):
    return df

# PART 1 GET DATA
def get_feats_labels_ids(df):
    ids = df['device_id']
    df = df.drop('device_id', axis=1)
    if 'device_class' in df:
        feats = df.drop('device_class', axis=1)
        labels = df['device_class']
    else:
        feats = df
        labels = []
    return feats, labels, ids

def get_data():
    #df = util.load_data_to_dataframe('dataset/train_split.json')
    #featurized_dataframe = create_features(df)
    #featurized_dataframe.to_csv('cache/train_split.csv', index=False)
    featurized_dataframe = filterout_mac_features(pd.read_csv('cache/train_split.csv'))
    train, val = features.split_train_val_data(featurized_dataframe)
    train_feats, train_labels, _ = get_feats_labels_ids(train)
    val_feats, val_labels, val_ids = get_feats_labels_ids(val)
    return train_feats, train_labels, val_feats, val_labels

def get_real_data():
    #df = util.load_data_to_dataframe('dataset/val_test_split.json')
    #unseen_test = create_features(df)
    #unseen_test.to_csv('cache/val_test_split.csv', index=False)
    unseen_test = filterout_mac_features(pd.read_csv('cache/val_test_split.csv'))
    train_feats, train_labels, _ = get_feats_labels_ids(unseen_test)
    return train_feats, train_labels

X, Y, X_test, Y_test = get_data()

# PART 2 FIT MODEL
k = 2

models = [None]*k
#models[0] = tree.DecisionTreeClassifier() 
models[1] = RandomForestClassifier(verbose=True, n_jobs=2, random_state=42, n_estimators=300)
models[0] = KerasModel()

kf = KFold(n_splits = k, shuffle = True, random_state = 2)
i = 0
for train_index, valid_index in kf.split(X):
    X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_val = Y.iloc[train_index], Y.iloc[valid_index]
    models[i].fit(X_train, y_train)
    
    print("predicting on kfold validation")
    val_predict = models[i].predict(X_val)
    print(f"f1s: {f1_score(y_val, val_predict, average='micro')}")
    i += 1
# PART 3 SAVE MODEL

# PART 4 EVALUATE
test_aggregate = np.zeros([len(X_test.index), 13])
for i in range(k):
    test_predict = models[i].predict_proba(X_test)
    test_aggregate += test_predict
test_aggregate = test_aggregate / k
test_ids = np.argmax(test_aggregate, axis=1)

classes = models[1].classes_
test_predictions = list(map(lambda x: classes[x], test_ids))
print(f"Average f1s: {f1_score(Y_test, test_predictions, average='micro')}")

# PART 5 EVALUATE ON UNSEEN
X_real, Y_real = get_real_data()

test_aggregate = np.zeros([len(X_real.index), 13])
for i in range(k):
    test_predict = models[i].predict_proba(X_real)
    test_aggregate += test_predict
test_aggregate = test_aggregate / k
test_ids = np.argmax(test_aggregate, axis=1)

test_predictions = list(map(lambda x: classes[x], test_ids))
print(f"Average f1s on unseen: {f1_score(Y_real, test_predictions, average='micro')}")

# PART 6 PREPARE SUBMISSION
def get_data_for_submitting():
    #df_test = util.load_data_to_dataframe('dataset/test.json')
    #prepared_df = create_features(df_test)
    #prepared_df.to_csv('cache/test.csv', index=False)
    prepared_df = filterout_mac_features(pd.read_csv('cache/test.csv'))
    test_feats, _, test_ids = get_feats_labels_ids(prepared_df)
    return test_feats, test_ids

def dump_for_submitting():
    X_submit, test_ids = get_data_for_submitting()
    print("predicting")
    test_aggregate = np.zeros([len(X_submit.index), 13])
    for i in range(k):
        test_predict = models[i].predict_proba(X_submit)
        test_aggregate += test_predict
    test_aggregate = test_aggregate / k
    class_ids = np.argmax(test_aggregate, axis=1)
    test_predictions = list(map(lambda x: classes[x], class_ids))

    with open("out/subm.csv", 'w') as f:
        f.write("Id,Predicted\n")
        for i in range(len(test_predictions)):
            f.write(f"{test_ids[i]},{test_predictions[i]}\n")
dump_for_submitting()
