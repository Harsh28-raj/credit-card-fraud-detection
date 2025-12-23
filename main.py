from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Credit Card Fraud Detection API")

# -----------------------------
# Load XGBoost model
# -----------------------------
model = xgb.XGBClassifier()
model.load_model("fraud_xgb_model.json")

# -----------------------------
# Define preprocessing IN CODE
# -----------------------------
NUM_COLS = ["amount", "time"]
CAT_COLS = ["merchant", "category"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
    ]
)

# -----------------------------
# Dummy fit (IMPORTANT)
# -----------------------------
# sklearn transformers need .fit()
dummy_df = pd.DataFrame({
    "amount": [0],
    "time": [0],
    "merchant": ["unknown"],
    "category": ["unknown"]
})

preprocessor.fit(dummy_df)

# -----------------------------
# Input schema
# -----------------------------
class Transaction(BaseModel):
    amount: float
    time: float
    merchant: str
    category: str

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(tx: Transaction):
    df = pd.DataFrame([tx.dict()])
    X = preprocessor.transform(df)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "fraud": bool(pred),
        "probability": float(prob)
    }

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def root():
    return {"status": "API running"}
