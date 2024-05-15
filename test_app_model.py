import pytest

from app_model import Predictor


@pytest.fixture
def predictor():
    predictor = Predictor()
    predictor.load_model()
    return predictor


def test_predict_positive(predictor):
    text = "This is a great movie!"
    prediction = predictor.predict(text)
    assert prediction == "Positive"  # Положительный отзыв


def test_predict_negative(predictor):
    text = "This movie was terrible, a waste of time."
    prediction = predictor.predict(text)
    assert prediction == 'Negative'  # Отрицательный отзыв


def test_predict_neutral(predictor):
    text = "Movie"
    prediction = predictor.predict(text)
    assert prediction == "Neutral"  # Нейтральный отзыв


# def test_predict_empty_string(predictor):
#     text = "19 years old"
#     prediction = predictor.predict(text)
#     assert prediction == "Irrelevant"  # Irrelevant
