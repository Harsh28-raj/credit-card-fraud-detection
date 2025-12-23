from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Credit Card Fraud Detection API")

model = joblib.load("fraud_xgb_model.pkl")

class Transaction(BaseModel):
    category: str
    gender: str
    state: str
    zip: int
    lat: float
    long: float
    city_pop: int
    merch_lat: float
    merch_long: float
    log_amt: float
    hour: int
    dayofweek: int
    is_weekend: int
    cust_merch_dist: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running"}

@app.post("/predict")
def predict(data: Transaction):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[0][1]
    return {
        "fraud_probability": float(prob),
        "prediction": "Fraud" if prob >= 0.3 else "Not Fraud"
    }
