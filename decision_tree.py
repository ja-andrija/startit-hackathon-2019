import util
from sklearn import tree
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from features import create_features_and_split, create_features
import matplotlib.pyplot as plt
import numpy as np

LABELS = ['AUDIO',
    'GAME_CONSOLE',
    'HOME_AUTOMATION',
    'IP_PHONE',
    'MEDIA_BOX',
    'MOBILE',
    'NAS',
    'PC',
    'PRINTER',
    'SURVEILLANCE',
    'TV',
    'VOICE_ASSISTANT',
    'GENERIC_IOT']

def get_feats_labels(df):
    feats = df.drop('device_class', axis=1)
    labels = df['device_class']
    return feats, labels

def get_data_for_submitting():
    df_train = util.load_data_to_dataframe('dataset/train.json')
    df_test = util.load_data_to_dataframe('dataset/test.json')
    train_feats, train_labels = get_feats_labels(create_features(df_train))
    test_feats, _ = get_feats_labels(create_features(df_test))
    return train_feats, train_labels, test_feats

def get_real_train_valid():
    df_train = util.load_data_to_dataframe('dataset/train_split_orig.json')
    df_val = util.load_data_to_dataframe('dataset/val_split_orig.json')
    train_feats, train_labels = get_feats_labels(create_features(df_train))
    val_feats, val_labels = get_feats_labels(create_features(df_val))
    return train_feats, train_labels, val_feats, val_labels

def get_data():
    df = util.load_data_to_dataframe('dataset/train_split_orig.json')
    train, val = create_features_and_split(df)
    train_feats, train_labels = get_feats_labels(train)
    val_feats, val_labels = get_feats_labels(val)
    return train_feats, train_labels, val_feats, val_labels

def plot_conf_matrix(cm):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.get_cmap("RdYlGn"))
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=LABELS, yticklabels=LABELS,
           title="Confusion matrix",
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

def main():
    model = tree.DecisionTreeClassifier()
    print("getting data")
    train_feats, train_labels, val_feats, val_labels = get_data()
    print("training")
    model.fit(train_feats, train_labels)
    print("predicting")
    val_predict = model.predict(val_feats)
    print(f"acc: {accuracy_score(val_labels, val_predict)}")
    print(f"f1s: {f1_score(val_labels, val_predict, average='micro')}")
    cm = confusion_matrix(val_labels, val_predict, labels=LABELS)

    [x for x in zip(train_feats['device_id'], val_labels, val_predict)]

    print(val_predict)

    plot_conf_matrix(cm)
    plt.show()
    tree.export_graphviz(model, out_file='tree.dot', feature_names = train_feats.columns, class_names = LABELS)

def train_and_dump_for_submitting():
    model = tree.DecisionTreeClassifier()
    print("getting data")
    train_feats, train_labels, test_feats = get_data_for_submitting()
    print("training")
    model.fit(train_feats, train_labels)
    print("predicting")
    test_predict = model.predict(test_feats)
    zip(test_feats['device_id'], test_predict)


main()