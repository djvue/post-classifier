import re

import joblib
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

nltk.download('stopwords', raise_on_error=True, quiet=True)
stop_words = set(stopwords.words('english'))


def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()


def remove_symbols(text):
    pattern = r'[^A-Za-z\s]'
    text = re.sub(pattern, '', text)
    text = ' '.join([word for word in text.split() if word.lower() not in stop_words])
    return text


if __name__ == '__main__':
    with open("data/vectorizer.joblib", "rb") as f:
        tfidf_vectorizer = joblib.load(f)
    with open("data/model.joblib", "rb") as f:
        model = joblib.load(f)

    df_test = pd.read_csv("data/test.csv")
    df_test.columns = ['ID', 'Category', 'Sentiment', 'Text']

    df_test['Text'] = df_test['Text'].apply(lambda x: remove_html_tags(x))
    df_test['Text'] = df_test['Text'].apply(lambda x: remove_symbols(x))

    X_test = df_test['Text']
    y_test = df_test.Sentiment

    tfidf_test = tfidf_vectorizer.transform(X_test)

    y_pred = model.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)

    print(f'Acc: {round(score * 100, 2)}%')

    print(confusion_matrix(y_test, y_pred, labels=['Neutral', 'Positive', 'Negative']))
