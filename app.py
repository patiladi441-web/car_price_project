import streamlit as st
import pandas as pd
import joblib
import numpy as np


import os

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))

st.title("🚗 Car Price Prediction")
name = st.selectbox("Brand", ["Maruti", "Hyundai", "Honda", "Tata"])
fuel = st.selectbox("Fuel", ["Petrol", "Diesel", "CNG"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])

km_driven = st.number_input("KM Driven")
car_age = st.number_input("Car Age")

mileage = st.number_input("Mileage")
engine = st.number_input("Max Engine CC")
max_power = 150
seats = st.number_input("Max Seats")

torque_nm = 200
torque_rpm = 4000


if st.button("Predict Price"):

    data = pd.DataFrame({
        'name': [name],
        'fuel': [fuel],
        'owner': [owner],
        'transmission': [transmission],
        'seller_type': [seller_type],
        'km_driven': [km_driven],
        'car_age': [car_age],
        'mileage': [mileage],
        'engine': [engine],
        'max_power': [max_power],
        'seats': [seats],
        'torque_nm': [torque_nm],
        'torque_rpm': [torque_rpm]
    })

    data_processed = preprocessor.transform(data)
    prediction = model.predict(data_processed)

    st.success(f"Average Estimated Price: ₹ {int(prediction[0])}")