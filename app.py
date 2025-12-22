import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("fraud_xgb_model.pkl")

st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud probability")

# User Inputs
category = st.selectbox("Category", [
    "gas_transport", "grocery_pos", "shopping_pos", "shopping_net",
    "entertainment", "food_dining", "misc_pos", "misc_net"
])

gender = st.selectbox("Gender", ["M", "F"])
state = st.text_input("State (e.g. CA, NY)", "CA")

zip_code = st.number_input("ZIP Code", min_value=0, value=12345)
lat = st.number_input("Customer Latitude", value=40.0)
long = st.number_input("Customer Longitude", value=-75.0)
city_pop = st.number_input("City Population", min_value=0, value=100000)

merch_lat = st.number_input("Merchant Latitude", value=41.0)
merch_long = st.number_input("Merchant Longitude", value=-74.0)

log_amt = st.number_input("Log Amount", value=3.5)
hour = st.slider("Transaction Hour", 0, 23, 12)
dayofweek = st.slider("Day of Week (0=Mon)", 0, 6, 3)
is_weekend = st.selectbox("Is Weekend?", [0, 1])
cust_merch_dist = st.number_input("Customer–Merchant Distance (km)", value=50.0)

# Create input DataFrame
input_df = pd.DataFrame([{
    "category": category,
    "gender": gender,
    "state": state,
    "zip": zip_code,
    "lat": lat,
    "long": long,
    "city_pop": city_pop,
    "merch_lat": merch_lat,
    "merch_long": merch_long,
    "log_amt": log_amt,
    "hour": hour,
    "dayofweek": dayofweek,
    "is_weekend": is_weekend,
    "cust_merch_dist": cust_merch_dist
}])

if st.button("Predict Fraud"):
    proba = model.predict_proba(input_df)[0][1]
    prediction = "🚨 FRAUD" if proba > 0.3 else "✅ Not Fraud"

    st.subheader("Result")
    st.write(f"**Fraud Probability:** `{proba:.6f}`")
    st.write(f"**Prediction:** {prediction}")
