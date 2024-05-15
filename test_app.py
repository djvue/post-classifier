import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_info(client):
    response = client.get("/_info")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_sentiment(client):
    response = client.post("/api/sentiment", json={"text": "positive text"})
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "sentiment" in data
    assert data["status"] == "ok"
    assert data["sentiment"] in ["Negative", "Neutral", "Positive"]
