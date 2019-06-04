from keras.models import Sequential
from keras.layers import Dense, Input
import numpy as np
from sklearn.base import BaseEstimator

CLASSES_TO_INT = {
    'AUDIO': 0,
    'GAME_CONSOLE': 1,
    'GENERIC_IOT':2,
    'HOME_AUTOMATION': 3,
    'IP_PHONE':4,
    'MEDIA_BOX':5,
    'MOBILE':6,
    'NAS':7,
    'PC':8,
    'PRINTER':9,
    'SURVEILLANCE':10,
    'TV':11,
    'VOICE_ASSISTANT':12
}

def labels_onehot(y):
    all_labels  = []
    for label in y:
        one_hot = len(CLASSES_TO_INT) * [0]
        index = label
        if type(label) is str:
            index = CLASSES_TO_INT[label]
        one_hot[index] = 1
        all_labels.append(one_hot)
    return np.array(all_labels)


class KerasModel(BaseEstimator):
    def __init__(self):
        a = Sequential()
        #a.add(Dense(2048, activation = 'tanh'))
        a.add(Dense(2048, activation = 'tanh'))
        a.add(Dense(1024, activation = 'tanh'))
        a.add(Dense(256, activation = 'tanh'))
        #a.add(Dense(256, activation = 'tanh'))
        a.add(Dense(13, activation='softmax'))
        self.model= a

    def fit(self, X, Y):
        y_map = labels_onehot(Y)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(x=X.values, y=y_map, epochs= 1)
    
    def predict(self, X):
        preds = self.model.predict(X)
        inverted_classes = {v:k for k, v in CLASSES_TO_INT.items()}
        ret_preds = []
        for pred in preds:
            ret_preds.append(inverted_classes[np.argmax(pred)])
        return ret_preds
    
    def predict_proba(self, X):
        return self.model.predict(X)

    def classes(self):
        return CLASSES_TO_INT.keys()

