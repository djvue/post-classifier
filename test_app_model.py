import pytest

from app_model import Predictor


@pytest.fixture
def predictor():
    predictor = Predictor()
    predictor.load_model()
    return predictor


def test_predict_positive(predictor):
    text = "Happy to be back doing what I love :))"
    prediction = predictor.predict(text)
    assert prediction == "Positive"  # Положительный отзыв


def test_predict_negative(predictor):
    text = "CSGO matchmaking is so full of closet hacking, it's a truly awful game."
    prediction = predictor.predict(text)
    assert prediction == 'Negative'  # Отрицательный отзыв


def test_predict_neutral(predictor):
    text = "Playing fifa with my girl.  She got her first goal against me and someone won’t shut up 😩"
    prediction = predictor.predict(text)
    assert prediction == "Neutral"  # Нейтральный отзыв


def test_predict_irrelevant(predictor):
    text = "Flip The Fuck Out!!!! Cyklon30001189  just joined the Kingdom on Mixer mixer.com/deduke #mixerPartner #mixer #streamer #Xbox #CallofDuty"
    prediction = predictor.predict(text)
    assert prediction == "Irrelevant"  # Irrelevant
