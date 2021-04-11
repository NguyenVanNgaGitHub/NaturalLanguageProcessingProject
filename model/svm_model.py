from sklearn.svm import SVC
from model.abstract_model import AbstractModel

class SVMModel(AbstractModel):
    def __init__(self):
        self.clf = SVC()

    def fit(self, X, y):
        self.clf.fit(X,y)

    def transform(self, X):
        return self.clf.predict(X)