import numpy as np
import pandas as pd
from dataset_preparation.util import device_names

class weighted_average_ensemble:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        probabilities = np.zeros((len(X.index), 13))
        for i, model in enumerate(self.models):
            probs = model.predict_proba(X)
            scaled_probabilities = np.multiply(probs, self.weights[i])
            probabilities += scaled_probabilities
        probabilities = probabilities / len(self.models)
        indicies = np.argmax(probabilities, axis=1)
        result = pd.Series(indicies)
        return result.apply(lambda x: device_names[x])