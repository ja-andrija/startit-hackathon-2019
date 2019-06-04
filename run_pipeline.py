from models.random_forest import RandomForest
import pandas as pd
from feature_engineering.features import get_data, dataset
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from models.keras_model import KerasModel
from models.weighted_average_ensemble import weighted_average_ensemble

invalidate_cache=False

# PART 1 GET DATA
X_train, Y_train = get_data(dataset.TRAIN_SPLIT, invalidate_cache)

# PART 2 FIT MODEL
k = 2

models = [None]*k
models[0] = KerasModel()
models[1] = RandomForest()

kf = KFold(n_splits = k, shuffle = True, random_state = 2)
i = 0
weights = [[0]*13]*k
for train_index, valid_index in kf.split(X_train):
    X, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
    Y, Y_val = Y_train.iloc[train_index], Y_train.iloc[valid_index]
    models[i].fit(X, Y)
    
    print("predicting on kfold validation")
    val_predict = models[i].predict(X_val)
    print(f"f1s: {f1_score(Y_val, val_predict, average='micro')}")
    weights[i] = f1_score(Y_val, val_predict, average=None)
    i += 1

# PART 3 FIT ENSEMBLE
ensemble = weighted_average_ensemble(models, weights)

# PART 4 EVALUATE ON TEST_SPLIT
X_test, Y_test = get_data(dataset.TEST_SPLIT, invalidate_cache)

#test our model on the test data
test_predictions = ensemble.predict(X_test)

print(f"Average f1s on unseen: {f1_score(Y_test, test_predictions, average='micro')}")

def dump_for_submitting():
    X_submit, test_ids = get_data(dataset.SUBMISSION)
    
    Y_submit = ensemble.predict(X_submit)

    with open("out/subm.csv", 'w') as f:
        f.write("Id,Predicted\n")
        for i in range(len(Y_submit)):
            f.write(f"{test_ids[i]},{Y_submit[i]}\n")
dump_for_submitting()
