import pytest

from train_model import *

TRAIN_DATA_URL = "https://raw.githubusercontent.com/djvue/ml-models-mirror/main/josuanicolas-sentiment-excercise/twitter_training.csv"
TEST_DATA_URL = "https://raw.githubusercontent.com/djvue/ml-models-mirror/main/josuanicolas-sentiment-excercise/twitter_validation.csv"


@pytest.fixture
def vectorizer():
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    return tfidf_vectorizer


def test_remove_html_tags():
    text = "<p>This is a paragraph with <b>bold</b> text.</p>"
    expected = "This is a paragraph with bold text."
    assert remove_html_tags(text) == expected


def test_remove_symbols():
    text = "symbols: @#$%^&*()."
    expected = "symbols"
    assert remove_symbols(text) == expected

    text_2 = "remove stuffs! &*%"
    expected_2 = "remove stuffs"
    assert remove_symbols(text_2) == expected_2

    text_3 = "no symbols here"
    expected_3 = "symbols"
    assert remove_symbols(text_3) == expected_3


def test_tfidf_vectorizer(vectorizer):
    X_train = ["This is a positive tweet", "This is a negative tweet"]
    tfidf_train = vectorizer.fit_transform(X_train)
    assert tfidf_train.shape == (2, 2)


def test_normal_dataset(vectorizer):
    if not os.path.isfile("data/train.csv"):
        os.makedirs("data", exist_ok=True)
        df_train = pd.read_csv(TRAIN_DATA_URL)
        df_train.to_csv("data/train.csv", index=False)
    if not os.path.isfile("data/test.csv"):
        os.makedirs("data", exist_ok=True)
        df_test = pd.read_csv(TEST_DATA_URL)
        df_test.to_csv("data/test.csv", index=False)

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

    tfidf_train = vectorizer.fit_transform(X_train)
    tfidf_test = vectorizer.transform(X_test)

    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(tfidf_train, y_train)

    y_pred = pac.predict(tfidf_test)
    score = accuracy_score(y_test, y_pred)
    assert score > 0.92
