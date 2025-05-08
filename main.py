from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

classifier = pipeline("sentiment-analysis", model="./model")

class TextData(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(data: TextData):
    result = classifier(data.text)
    return result
