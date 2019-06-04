from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator

class RandomForest(BaseEstimator):
    def __init__(self):
        # Set the parameters by cross-validation
        self.tuned_parameters = {'n_estimators': [50, 100, 200]}

        self.clf = GridSearchCV(RandomForestClassifier(), self.tuned_parameters, cv=5)

    def fit(self, X, Y):
        self.clf.fit(X, Y)
        self.estimator = self.clf.best_estimator_

    def predict(self, X):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def classes(self):
        return self.estimator.classes_