import util
from sklearn import tree
from sklearn.metrics import accuracy_score
from features import create_features_and_split
from sklearn import ensemble
from sklearn import linear_model
from sklearn.ensemble import GradientBoostingClassifier

def get_feats_labels(df):
    feats = df.drop('device_class', axis=1)
    labels = df['device_class']
    return feats, labels

def get_data():
    df = util.load_data_to_dataframe('dataset/train_split_orig.json')
    train, val = create_features_and_split(df)
    train_feats, train_labels = get_feats_labels(train)
    val_feats, val_labels = get_feats_labels(val)
    return train_feats, train_labels, val_feats, val_labels

def main():
    train_feats, train_labels, val_feats, val_labels = get_data()
    
    gb = GradientBoostingClassifier(verbose=True, n_estimators=100, learning_rate = 0.05, max_features=20, max_depth = 20, random_state = 0)
    gb.fit(train_feats, train_labels)

    val_predict = gb.predict(val_feats)
    print(accuracy_score(val_labels, val_predict))

main()