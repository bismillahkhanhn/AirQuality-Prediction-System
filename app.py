import streamlit as st
import joblib
import numpy as np

# Load trained model
lr_model = joblib.load("linear_regression_pm25.pkl")

st.title("🌬️ PM2.5 Prediction App (Linear Regression)")

# Sliders for input
PM10 = st.slider("PM10", 0, 600, 100)
NO = st.slider("NO", 0, 500, 50)
NO2 = st.slider("NO2", 0, 500, 30)
NOx = st.slider("NOx", 0, 500, 60)
NH3 = st.slider("NH3", 0, 200, 20)
CO = st.slider("CO", 0.0, 50.0, 2.0)
SO2 = st.slider("SO2", 0, 300, 15)
O3 = st.slider("O3", 0, 300, 40)
Benzene = st.slider("Benzene", 0.0, 50.0, 5.0)
Toluene = st.slider("Toluene", 0.0, 50.0, 4.0)
Xylene = st.slider("Xylene", 0.0, 50.0, 3.0)

if st.button("Predict PM2.5"):
    features = np.array([[PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]])
    prediction = lr_model.predict(features)[0]

    # AQI Category function
    def pm25_to_aqi_category(pm):
        if pm <= 12: return "Good"
        elif pm <= 35.4: return "Moderate"
        elif pm <= 55.4: return "Unhealthy (Sensitive)"
        elif pm <= 150.4: return "Unhealthy"
        elif pm <= 250.4: return "Very Unhealthy"
        else: return "Hazardous"

    aqi_cat = pm25_to_aqi_category(prediction)
    
    st.success(f"✅ Predicted PM2.5: {prediction:.2f}")
    st.info(f"🌡️ AQI Category: {aqi_cat}")
