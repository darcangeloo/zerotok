from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.classify import classify_document

app = FastAPI()

class ClassificationRequest(BaseModel):
    document: str
    categories: list[str]

@app.post("/classify")
def classify(request: ClassificationRequest):
    probabilities = classify_document(
        request.document,
        request.categories
    )

    category = max(probabilities, key=probabilities.get)

    return {
        "category": category,
        "confidence": probabilities[category],
        "probabilities": probabilities
    }

app.mount("/frontend", StaticFiles(directory="frontend"), name="ui")

@app.get("/")
def dashboard():
    return FileResponse("frontend/index.html")