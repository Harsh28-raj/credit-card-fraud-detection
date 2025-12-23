from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import numpy as np
import joblib

app = FastAPI()

# Load preprocessor (OK as joblib)
preprocessor = joblib.load("preprocessor.joblib")

# Load XGBoost Booster (NOT XGBClassifier)
model = xgb.Booster()
model.load_model("fraud_xgb_model.json")

# -------- Input schema --------
class FraudInput(BaseModel):
    amount: float
    time: float
    merchant: str
    category: str

@app.get("/")
def health():
    return {"status": "API running"}

@app.post("/predict")
def predict(data: FraudInput):

    # Convert input to DataFrame-like format
    input_dict = {
        "amount": [data.amount],
        "time": [data.time],
        "merchant": [data.merchant],
        "category": [data.category]
    }

    X = preprocessor.transform(
        joblib.load("columns.joblib").__class__(input_dict)
        if False else preprocessor.transform(
            __import__("pandas").DataFrame(input_dict)
        )
    )

    dmatrix = xgb.DMatrix(X)

    prob = model.predict(dmatrix)[0]
    fraud = prob > 0.5

    return {
        "fraud": bool(fraud),
        "probability": float(prob)
    }
