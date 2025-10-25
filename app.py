import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load trained Linear Regression model
# Make sure linear_regression_pm25.pkl is in the same folder
# -------------------------
lr_model = joblib.load("linear_regression_pm25.pkl")

# -------------------------
# App Title & Description
# -------------------------
st.set_page_config(page_title="🌬️ PM2.5 Prediction App", layout="wide")
st.title("🌬️ PM2.5 Prediction App (Linear Regression)")
st.markdown("""
Adjust the pollutant values in the sidebar and click **Predict PM2.5**.  
The app will show the predicted PM2.5 value along with its AQI category.
""")

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Input Pollutants & City")

city = st.sidebar.selectbox("Select City", ["Delhi", "Mumbai", "Chennai", "Kolkata", "Bangalore"])

PM10 = st.sidebar.slider("PM10", 0, 600, 100)
NO = st.sidebar.slider("NO", 0, 500, 50)
NO2 = st.sidebar.slider("NO2", 0, 500, 30)
NOx = st.sidebar.slider("NOx", 0, 500, 60)
NH3 = st.sidebar.slider("NH3", 0, 200, 20)
CO = st.sidebar.slider("CO", 0.0, 50.0, 2.0)
SO2 = st.sidebar.slider("SO2", 0, 300, 15)
O3 = st.sidebar.slider("O3", 0, 300, 40)
Benzene = st.sidebar.slider("Benzene", 0.0, 50.0, 5.0)
Toluene = st.sidebar.slider("Toluene", 0.0, 50.0, 4.0)
Xylene = st.sidebar.slider("Xylene", 0.0, 50.0, 3.0)

# -------------------------
# Prediction & Visualization
# -------------------------
if st.button("Predict PM2.5"):

    # Features for prediction
    features = np.array([[PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]])
    prediction = lr_model.predict(features)[0]

    # Convert PM2.5 to AQI Category
    def pm25_to_aqi_category(pm):
        if pm <= 12: return "Good", "green"
        elif pm <= 35.4: return "Moderate", "yellow"
        elif pm <= 55.4: return "Unhealthy (Sensitive)", "orange"
        elif pm <= 150.4: return "Unhealthy", "red"
        elif pm <= 250.4: return "Very Unhealthy", "purple"
        else: return "Hazardous", "maroon"

    aqi_cat, color = pm25_to_aqi_category(prediction)

    # Display results
    st.success(f"✅ Predicted PM2.5: {prediction:.2f} µg/m³")
    st.markdown(f"<h3 style='color:{color};'>🌡️ AQI Category: {aqi_cat}</h3>", unsafe_allow_html=True)

    # Bar chart visualization
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(['PM2.5'], [prediction], color=color)
    ax.set_ylim(0, max(prediction+50, 200))
    ax.set_ylabel("PM2.5 Value (µg/m³)")
    ax.set_title(f"Predicted PM2.5 for {city}")
    st.pyplot(fig)

# -------------------------
# Footer / Notes
# -------------------------
st.markdown("---")
st.markdown("""
**Note:** This prediction is based on a Linear Regression model trained on historical pollutant data.  
Values are indicative and should not replace official AQI reports.
""")
