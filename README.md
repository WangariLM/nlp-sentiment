# Sentiment Analysis API

A fine-tuned DistilBERT model for binary sentiment classification, served as a REST API, containerized with Docker, and tested with an automated CI pipeline.

## Results

- **Model:** DistilBERT (distilbert-base-uncased), fine-tuned on SST-2
- **Accuracy:** 90.83%
- **F1 Score:** 0.9081
- Experiment tracked with MLflow (params, metrics, model artifacts)

## Architecture
Colab (GPU)              HuggingFace Hub          Local / CI (CPU)

─────────────             ───────────────           ──────────────────

Fine-tune DistilBERT  →   Model hosted publicly  →  FastAPI serves predictions

MLflow experiment            (256MB weights)         Docker containerizes the API

tracking                                          GitHub Actions runs tests + build

## Tech Stack

Python, PyTorch, HuggingFace Transformers, FastAPI, Docker, GitHub Actions, MLflow, pytest

## Project Structure
nlp-sentiment/

├── app/

│   ├── main.py           # FastAPI application

│   └── init.py

├── models/                # fine-tuned model (gitignored, pulled from HF Hub)

├── tests/

│   └── test_main.py      # automated test suite

├── notebooks/             # training notebook

├── .github/workflows/

│   └── ci.yml             # CI: test + build on every push

├── download_model.py       # pulls model from HuggingFace Hub

├── Dockerfile

├── requirements.txt

└── pytest.ini

## Running Locally

**1. Clone and set up:**
```bash
git clone https://github.com/WangariLM/nlp-sentiment.git
cd nlp-sentiment
pip install -r requirements.txt
```

**2. Download the model:**
```bash
python download_model.py
```

**3. Run the API:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Running with Docker

**Option A : pull the pre-built image:**
```bash
docker run -d -p 8000:8000 wangari/sentiment-api:v1
```

**Option B : build it yourself:**
```bash
docker build -t sentiment-api:v1 .
docker run -d -p 8000:8000 sentiment-api:v1
```

## API Usage

**Health check:**
```bash
curl http://localhost:8000/health
```

**Predict sentiment:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely wonderful"}'
```

Response:
```json
{
  "text": "This movie was absolutely wonderful",
  "sentiment": "POSITIVE",
  "confidence": 0.9998,
  "probabilities": {"negative": 0.0002, "positive": 0.9998}
}
```

Interactive docs available at `http://localhost:8000/docs`.

## Testing

```bash
pytest tests/ -v
```

6 tests covering endpoint health, positive/negative prediction accuracy, input validation, and response schema.

## CI/CD

Every push to `main` automatically:
1. Downloads the model from HuggingFace Hub
2. Runs the full test suite
3. Builds the Docker image

See `.github/workflows/ci.yml`.

## Links

- Model: https://huggingface.co/WangariLM/distilbert-sst2-sentiment
- Docker image: https://hub.docker.com/r/wangarilm/sentiment-api

## Future Improvements

- Dynamic padding for faster inference
- Load model from HuggingFace Hub at container startup instead of baking weights into the image (would reduce image size and decouple model updates from image rebuilds)
- Additional fine-tuning experiments (learning rate, max_length, alternative base models) tracked via MLflow
