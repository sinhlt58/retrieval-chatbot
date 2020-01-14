import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd
from pandas import DataFrame


class Model:

    def __init__(self):
        pass

    def train(self, **kwargs):
        raise NotImplementedError(
            "Not implement train function"
        )

    def predict(self, **kwargs):
        raise NotImplementedError(
            "Not implement predict function"
        )

    def save(self, **kwargs):
        raise NotImplementedError(
            "Not implement save function"
        )

    def load(self, **kwargs):
        raise NotImplementedError(
            "Not implement load function"
        )


class SklearnModel(Model):

    def __init__(self, name):
        self.name = name

    def train(self, texts, y):
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2)
        )
        # We simply concat the context and the response to
        # a paragraph
        X = vectorizer.fit_transform(texts)

        # build the model
        lg = LogisticRegression()
        parameters = {
            "C": [0.1, 1, 10, 100],
            "max_iter": [1000],
        }
        model = GridSearchCV(lg, parameters)
        model.fit(X, y)
        print ("resutls: ", model.cv_results_)

        self.vectorizer = vectorizer
        self.model = model

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X), self.model.predict_proba(X)

    def save(self, data_dir):
        with open(f"{data_dir}/{self.name}.vec", "wb") as f:
            pickle.dump(self.vectorizer, f)
        with open(f"{data_dir}/{self.name}.cls", "wb") as f:
            pickle.dump(self.model, f)

    def load(self, data_dir):
        with open(f"{data_dir}/{self.name}.vec", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(f"{data_dir}/{self.name}.cls", "rb") as f:
            self.model = pickle.load(f)


class DSTC7FirstPlace(Model):

    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass

    def save(self):
        pass

    def load(self):
        pass
