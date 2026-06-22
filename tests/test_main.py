from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Sentiment Analysis API is running"}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_path" in data
    assert "device" in data


def test_predict_positive():
    response = client.post(
        "/predict",
        json={"text": "This movie was absolutely wonderful"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "POSITIVE"
    assert data["confidence"] > 0.5


def test_predict_negative():
    response = client.post(
        "/predict",
        json={"text": "This was a complete waste of time and money"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "NEGATIVE"
    assert data["confidence"] > 0.5


def test_predict_empty_text():
    response = client.post(
        "/predict",
        json={"text": ""}
    )
    assert response.status_code == 400


def test_predict_response_structure():
    response = client.post(
        "/predict",
        json={"text": "I really enjoyed this"}
    )
    data = response.json()
    assert "text" in data
    assert "sentiment" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "negative" in data["probabilities"]
    assert "positive" in data["probabilities"]
