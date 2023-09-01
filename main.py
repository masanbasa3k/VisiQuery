from model import model_pipeline

from fastapi import FastAPI, UploadFile
from typing import Union
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Use /docs to get a prediction"}

@app.post("/predict")
def predict(text: str, image: UploadFile):
    content = image.file.read()

    image = Image.open(io.BytesIO(content))

    result = model_pipeline(text, image)
    return {"result": result}
