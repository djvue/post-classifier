import joblib

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.feature_extraction.text import TfidfVectorizer


class Predictor:
    model: PassiveAggressiveClassifier
    tfidf_vectorizer: TfidfVectorizer

    def __init__(self):
        pass

    def load_model(self):
        with open("data/vectorizer.joblib", "rb") as f:
            self.tfidf_vectorizer = joblib.load(f)
        with open("data/model.joblib", "rb") as f:
            self.model = joblib.load(f)

    def predict(self, text: str):
        text = self.tfidf_vectorizer.transform([text])[0]

        y_pred = self.model.predict(text)

        return y_pred[0]
