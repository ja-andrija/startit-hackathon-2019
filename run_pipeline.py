import util
from features import create_features_and_split, create_features
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier

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
    df = util.load_data_to_dataframe('dataset/train_split_orig.json')
    train, val = create_features_and_split(df)
    train_feats, train_labels, _ = get_feats_labels_ids(train)
    val_feats, val_labels, val_ids = get_feats_labels_ids(val)
    return train_feats, train_labels, val_feats, val_labels

def get_real_data():
    df = util.load_data_to_dataframe('dataset/val_split_orig.json')
    unseen_test = create_features(df)
    train_feats, train_labels, _ = get_feats_labels_ids(unseen_test)
    return train_feats, train_labels

X, Y, X_test, Y_test = get_data()

# PART 2 FIT MODEL
k = 5

models = [None]*k
models[0] = tree.DecisionTreeClassifier() 
models[1] = GradientBoostingClassifier(verbose=True, n_estimators=20, learning_rate = 0.1, max_features=20, max_depth = 10, random_state = 0)
models[2] = RandomForestClassifier(verbose=True, n_jobs=2, random_state=0)
models[3] = AdaBoostClassifier(n_estimators=50, learning_rate=0.1)
models[4] = tree.DecisionTreeClassifier()

kf = KFold(n_splits = k, shuffle = True, random_state = 2)
i = 0
for train_index, valid_index in kf.split(X):
    print("TRAIN:", train_index, "VALIDATE:", valid_index)
    X_train, X_val = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_val = Y.iloc[train_index], Y.iloc[valid_index]
    models[i].fit(X_train, y_train)
    
    print("predicting")
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

classes = models[0].classes_
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

classes = models[0].classes_
test_predictions = list(map(lambda x: classes[x], test_ids))
print(f"Average f1s on unseen: {f1_score(Y_real, test_predictions, average='micro')}")

# PART 6 PREPARE SUBMISSION
def get_data_for_submitting():
    df_test = util.load_data_to_dataframe('dataset/test.json')
    test_feats, _, test_ids = get_feats_labels_ids(create_features(df_test))
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
    classes = models[0].classes_
    test_predictions = list(map(lambda x: classes[x], class_ids))

    with open("out/subm.csv", 'w') as f:
        f.write("Id,Predicted\n")
        for i in range(len(test_predictions)):
            f.write(f"{test_ids[i]},{test_predictions[i]}\n")
dump_for_submitting()
