from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import xgboost as xgb
import numpy as np

app = FastAPI(title="Credit Card Fraud Detection API")

# Load preprocessor
preprocessor = joblib.load("preprocessor.joblib")

# Load XGBoost model (JSON – safe)
model = xgb.XGBClassifier()
model.load_model("fraud_xgb_model.json")


class Transaction(BaseModel):
    features: list[float]


@app.get("/")
def home():
    return {"status": "API running successfully"}


@app.post("/predict")
def predict(data: Transaction):
    X = np.array(data.features).reshape(1, -1)
    X_transformed = preprocessor.transform(X)
    prob = model.predict_proba(X_transformed)[0][1]

    return {
        "fraud_probability": float(prob),
        "fraud": bool(prob > 0.5)
    }
