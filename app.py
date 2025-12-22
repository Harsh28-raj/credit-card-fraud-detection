import streamlit as st
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np


st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load("preprocessor.joblib")
    model = xgb.XGBClassifier()
    model.load_model("fraud_xgb_model.json")
    return preprocessor, model

preprocessor, model = load_artifacts()

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud probability")

# ---- INPUTS ----
col1, col2, col3 = st.columns(3)

with col1:
    credit_usage = st.slider("Credit Usage (%)", 0, 100, 30)
    age = st.number_input("Age", 18, 100, 35)

with col2:
    income = st.number_input("Monthly Income", 1000, 500000, 50000)
    active_accounts = st.number_input("Active Credit Accounts", 0, 20, 5)

with col3:
    late_30 = st.number_input("Late Payments (30–59 days)", 0, 10, 0)
    late_60 = st.number_input("Late Payments (60–89 days)", 0, 10, 0)
    late_90 = st.number_input("Late Payments (90+ days)", 0, 10, 0)

if st.button("🔍 Predict Fraud Risk"):
    input_df = pd.DataFrame([{
        "credit_usage": credit_usage,
        "age": age,
        "income": income,
        "active_accounts": active_accounts,
        "late_30": late_30,
        "late_60": late_60,
        "late_90": late_90
    }])

    X = preprocessor.transform(input_df)
    proba = model.predict_proba(X)[0][1]

    st.subheader("Result")
    st.metric("Fraud Probability", f"{proba:.2%}")

    if proba > 0.7:
        st.error("🚨 High Risk Transaction")
    elif proba > 0.4:
        st.warning("⚠️ Medium Risk")
    else:
        st.success("✅ Low Risk")
