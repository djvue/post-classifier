from fastapi import FastAPI, Request
from app_model import Predictor
from pydantic import BaseModel


app = FastAPI()

predictor = Predictor()
predictor.load_model()


class SentimentIn(BaseModel):
    text: str


@app.get("/_info")
def info():
    return {"status": "ok"}


@app.post("/api/sentiment")
def process(
    request: Request,
    sentiment_in: SentimentIn,
):
    sentiment = predictor.predict(sentiment_in.text)

    return {"status": "ok", "sentiment": sentiment}