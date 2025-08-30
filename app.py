# app.py
import streamlit as st
import pickle
import numpy as np
import os

# -------------------------
# Load trained model
# -------------------------
model_file = r"C:\Users\Darnish S\OneDrive\Desktop\tripfare_project\scripts\random_forest_model.pkl"


# Check if the model file exists
if not os.path.exists(model_file):
    st.error(f"Model file not found: {model_file}")
    st.stop()

# Load the model
with open(model_file, "rb") as f:
    model = pickle.load(f)

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="NYC Taxi Trip Duration Predictor", layout="centered")
st.title("🚖 NYC Taxi Trip Duration Predictor")
st.markdown("Predict the estimated trip duration (in minutes) based on trip details.")

# Two-column layout for inputs
col1, col2 = st.columns(2)

with col1:
    vendor_id = st.selectbox("Vendor ID", [1, 2])
    passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, value=1)
    rate_code = st.selectbox("Rate Code", [1, 2, 3, 4, 5, 6])
    payment_type = st.selectbox("Payment Type", [1, 2, 3, 4, 5, 6])
    store_and_fwd_flag = st.selectbox("Store and Forward Flag", ["N", "Y"])

with col2:
    pickup_longitude = st.number_input("Pickup Longitude", value=-73.985428, format="%.6f")
    pickup_latitude = st.number_input("Pickup Latitude", value=40.748817, format="%.6f")
    dropoff_longitude = st.number_input("Dropoff Longitude", value=-73.985428, format="%.6f")
    dropoff_latitude = st.number_input("Dropoff Latitude", value=40.748817, format="%.6f")

# Encode categorical variable
store_flag = 1 if store_and_fwd_flag == "Y" else 0

# Prepare input features
features = np.array([[vendor_id, passenger_count,
                      pickup_longitude, pickup_latitude,
                      rate_code,
                      dropoff_longitude, dropoff_latitude,
                      payment_type,
                      store_flag]])

# Prediction button
if st.button("Predict Trip Duration"):
    try:
        prediction = model.predict(features)[0]
        st.success(f"🕒 Estimated Trip Duration: {prediction:.2f} minutes")
    except Exception as e:
        st.error(f"Error predicting trip duration: {e}")
