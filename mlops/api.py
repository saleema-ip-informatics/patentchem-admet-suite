import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from predict import predict

app = FastAPI(
    title="EGFR QSAR Learning API",
    description="Small-dataset demonstration; not validated for research decisions.",
)

model_setting = os.environ.get("MODEL_PATH")
if not model_setting:
    raise RuntimeError("Set MODEL_PATH before starting the API.")

MODEL_PATH = Path(model_setting).resolve()
if not MODEL_PATH.is_file():
    raise RuntimeError("Model file does not exist.")


class PredictionRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=10000)


@app.post("/predict")
def predict_smiles(request: PredictionRequest):
    try:
        return predict(MODEL_PATH, request.smiles)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
