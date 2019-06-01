import util
from sklearn import tree
from sklearn.metrics import accuracy_score
from features import create_features_and_split


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
    model = tree.DecisionTreeClassifier()
    model.fit(train_feats, train_labels)
    val_predict = model.predict(val_feats)
    print(accuracy_score(val_labels, val_predict))

main()