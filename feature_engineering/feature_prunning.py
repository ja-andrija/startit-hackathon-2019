import util
import features
from features import create_features_and_split, create_features
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns
import matplotlib.pyplot as plt

import eli5
from eli5.sklearn import PermutationImportance

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
    df = util.load_data_to_dataframe('dataset/train_split.json')
    train, val = create_features_and_split(df)
    train_feats, train_labels, _ = get_feats_labels_ids(train)
    val_feats, val_labels, val_ids = get_feats_labels_ids(val)
    return train_feats, train_labels, val_feats, val_labels

def get_real_data():
    df = util.load_data_to_dataframe('dataset/val_test_split.json')
    unseen_test = create_features(df)
    train_feats, train_labels, _ = get_feats_labels_ids(unseen_test)
    return train_feats, train_labels

'''df = util.load_data_to_dataframe('dataset/train_split.json')
F = create_features(df)
X, Y, _ = get_feats_labels_ids(F)'''

def compute_feature_significance(X, Y):
    bestfeatures = SelectKBest(score_func=chi2, k='all')
    fit = bestfeatures.fit(X,Y)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X.columns)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  #naming the dataframe columns
    featureScores.to_csv('feature_importance.csv')
    #print(featureScores.nlargest(100,'Score'))  #print 10 best features

#compute_feature_significance(X, Y)

#feature_scores = pd.read_csv('feature_importance.csv')
#top_features=feature_scores.nlargest(20,'Score')

#top_features_list = top_features['Specs'].values.tolist()

'''raw = util.load_data_to_dataframe('dataset/train_split.json')
original_df = create_features(raw)

df = original_df.drop(['device_class', 'device_id'], axis=1)

corr = df.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = df.columns[columns]
data = original_df[selected_columns.values.tolist() + ['device_class', 'device_id']]
data.to_csv('selected.csv')

print("DONE")'''

train_df = pd.read_csv('cache/train_split.csv')
X_train, y_train, _ = get_feats_labels_ids(train_df)

val_df = pd.read_csv('cache/val_test_split.csv')
X_test, y_test, _ = get_feats_labels_ids(train_df)

## Import the random forest model.
from sklearn.ensemble import RandomForestClassifier 
## This line instantiates the model. 
rf = RandomForestClassifier(verbose=True, n_jobs=2, random_state=42, n_estimators=300, max_depth=5)
## Fit the model on your training data.
rf.fit(X_train, y_train) 
## And score it on your testing data.
rf.score(X_test, y_test)

feature_importances = pd.DataFrame(rf.feature_importances_,index = X_train.columns,columns=['importance']).sort_values('importance',ascending=False)
feature_importances.to_csv("random_forest_importances.csv")
print(feature_importances)