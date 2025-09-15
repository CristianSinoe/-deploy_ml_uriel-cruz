# app.py
from typing import List, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel, Field, conint, confloat
import pandas as pd
import joblib
import os

# CORS (si lo requieres)
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = os.getenv("MODEL_PATH", "model/diabetes-model-v1.joblib")
payload = joblib.load(MODEL_PATH)
pipeline = payload["pipeline"]
THRESHOLD = float(payload["threshold"])
FEATURE_ORDER = payload["feature_order"]

app = FastAPI(title="Diabetes Risk API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# ==== Schemas ====
# Notas de tipos: age/bmi/HbA1c_level/blood_glucose_level son floats; hypertension/heart_disease son 0/1

class InputData(BaseModel):
    gender: str = Field(..., example="Female")                  # Male, Female, Other (manejado por OneHotEncoder ignorando desconocidos)
    age: confloat(ge=0, le=130) = Field(..., example=54.0)
    hypertension: conint(ge=0, le=1) = Field(..., example=0)
    heart_disease: conint(ge=0, le=1) = Field(..., example=1)
    smoking_history: str = Field(..., example="never")          # never, No Info, current, former, etc.
    bmi: confloat(gt=0, le=100) = Field(..., example=27.32)
    HbA1c_level: confloat(gt=0, le=20) = Field(..., example=6.6)
    blood_glucose_level: confloat(gt=0, le=1000) = Field(..., example=140)

class OutputData(BaseModel):
    score: float                      # probabilidad de clase positiva (diabetes = 1)

@app.get("/")
def root():
    return {"message": "Diabetes Risk API up", "threshold": THRESHOLD}

def to_dataframe(item: Union[InputData, List[InputData]]) -> pd.DataFrame:
    if isinstance(item, list):
        df = pd.DataFrame([i.dict() for i in item])
    else:
        df = pd.DataFrame([item.dict()])
    # Asegurar orden de columnas esperado por el pipeline
    return df[FEATURE_ORDER]

@app.post("/score", response_model=Union[OutputData, List[OutputData]])
def score(data: Union[InputData, List[InputData]]):
    df = to_dataframe(data)
    proba = pipeline.predict_proba(df)[:, 1]

    if isinstance(data, list):
        return [OutputData(score=float(p)) for p in proba]
    else:
        return OutputData(score=float(proba[0]))

