from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

app = FastAPI(
    title="Sentiment Analysis API",
    description="DistilBERT fine-tuned on SST-2",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "distilbert-sst2-final")
MODEL_PATH = os.path.abspath(MODEL_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()
print(f"Model loaded successfully on: {device}")

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict

@app.get("/")
def root():
    return {"message": "Sentiment Analysis API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_path": MODEL_PATH, "device": str(device)}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=64,
        padding="max_length"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = probabilities.argmax().item()
        confidence = probabilities.max().item()

    return PredictResponse(
        text=request.text,
        sentiment=LABELS[predicted_class],
        confidence=round(confidence, 4),
        probabilities={
            "negative": round(probabilities[0][0].item(), 4),
            "positive": round(probabilities[0][1].item(), 4)
        }
    )
