
import re

import joblib
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
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
    df_train = pd.read_csv("data/train.csv")
    df_train.columns = ['ID', 'Category', 'Sentiment', 'Text']

    df_test = pd.read_csv("data/test.csv")
    df_test.columns = ['ID', 'Category', 'Sentiment', 'Text']

    df_train = df_train.dropna()

    # Clean text column
    df_train['Text'] = df_train['Text'].apply(lambda x: remove_html_tags(x))
    df_train['Text'] = df_train['Text'].apply(lambda x: remove_symbols(x))

    df_test['Text'] = df_test['Text'].apply(lambda x: remove_html_tags(x))
    df_test['Text'] = df_test['Text'].apply(lambda x: remove_symbols(x))

    X_train = df_train['Text']
    y_train = df_train.Sentiment

    X_test = df_test['Text']
    y_test = df_test.Sentiment

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

    tfidf_train = tfidf_vectorizer.fit_transform(X_train)
    tfidf_test = tfidf_vectorizer.transform(X_test)

    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    # tfdf_first = tfidf_test[0]
    # first_pred = pac.predict(tfdf_first)

    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)

    print(f'Acc: {round(score * 100, 2)}%')

    print(confusion_matrix(y_test, y_pred, labels=['Neutral', 'Positive', 'Negative']))

    with open("data/vectorizer.joblib", "wb") as f:
        joblib.dump(tfidf_vectorizer, f)
    with open("data/model.joblib", "wb") as f:
        joblib.dump(pac, f)
