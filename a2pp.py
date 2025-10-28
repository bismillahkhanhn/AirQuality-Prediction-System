# ==============================
# 🌍 ENHANCED AIR QUALITY PREDICTION SYSTEM (Streamlit)
# ==============================

import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 🌈 PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="🌿 Air Quality Prediction System", layout="centered", page_icon="🌏")

# -------------------------------
# 🎨 TITLE SECTION
# -------------------------------
st.markdown("""
<h1 style='text-align:center; color:#2E7D32;'>🌿 Air Quality Prediction System</h1>
<p style='text-align:center; font-size:18px; color:#555;'>
Predict PM2.5 Levels and AQI Category using Machine Learning 🤖<br>
Trained with real Indian city air-quality data 🇮🇳
</p>
""", unsafe_allow_html=True)

# -------------------------------
# 🧠 LOAD MODEL
# -------------------------------
try:
    lr_model = joblib.load("linear_regression_pm25.pkl")
    st.success("✅ Model Loaded Successfully!")
except Exception as e:
    st.error("❌ Could not load model. Please check 'linear_regression_pm25.pkl' file.")
    st.stop()

st.markdown("---")

# -------------------------------
# 🏙️ USER INPUTS
# -------------------------------
city = st.text_input("🏙️ Enter City Name", placeholder="e.g., Delhi, Mumbai, Chennai")

col1, col2 = st.columns(2)
with col1:
    PM10 = st.number_input("PM10 (Coarse Particles µg/m³)", value=100.0)
    NO = st.number_input("NO (Nitric Oxide µg/m³)", value=50.0)
    NO2 = st.number_input("NO₂ (Nitrogen Dioxide µg/m³)", value=30.0)
    NOx = st.number_input("NOx (Nitrogen Oxides µg/m³)", value=60.0)
    NH3 = st.number_input("NH₃ (Ammonia µg/m³)", value=20.0)
with col2:
    CO = st.number_input("CO (Carbon Monoxide mg/m³)", value=2.0)
    SO2 = st.number_input("SO₂ (Sulfur Dioxide µg/m³)", value=15.0)
    O3 = st.number_input("O₃ (Ozone µg/m³)", value=40.0)
    Benzene = st.number_input("Benzene (C₆H₆ µg/m³)", value=5.0)
    Toluene = st.number_input("Toluene (C₇H₈ µg/m³)", value=4.0)
    Xylene = st.number_input("Xylene (C₈H₁₀ µg/m³)", value=3.0)

# -------------------------------
# 🚀 PREDICTION
# -------------------------------
if st.button("🚀 Predict Air Quality"):
    if not city.strip():
        st.error("❌ Please enter a valid city name!")
        st.stop()

    # Prepare input
    features = np.array([[PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]])
    prediction = lr_model.predict(features)[0]

    # AQI classification
    def pm25_to_aqi_category(pm):
        if pm <= 12: 
            return "Good", "green", "😊 Excellent! Air quality is clean and fresh."
        elif pm <= 35.4: 
            return "Moderate", "yellow", "🙂 Acceptable air quality, minor risk to sensitive individuals."
        elif pm <= 55.4: 
            return "Unhealthy for Sensitive Groups", "orange", "😷 Some people with lung conditions may feel discomfort."
        elif pm <= 150.4: 
            return "Unhealthy", "red", "⚠️ Everyone may experience health effects; limit outdoor activities."
        elif pm <= 250.4: 
            return "Very Unhealthy", "purple", "☠️ Serious health risks, avoid outdoor exposure."
        else: 
            return "Hazardous", "maroon", "🚨 Dangerous levels! Immediate health warnings issued."

    aqi_cat, color, desc = pm25_to_aqi_category(prediction)

    # Results Display
    st.markdown(f"<h2>Predicted PM2.5: <span style='color:#1565C0;'>{prediction:.2f} µg/m³</span></h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color:{color};'>AQI Category: {aqi_cat}</h3>", unsafe_allow_html=True)
    st.info(desc)

    # -------------------------------
    # 📊 AQI SCALE VISUALIZATION
    # -------------------------------
    fig, ax2 = plt.subplots(figsize=(6, 2))
    aqi_levels = [12, 35.4, 55.4, 150.4, 250.4, 500]
    aqi_colors = ['green', 'yellow', 'orange', 'red', 'purple', 'maroon']
    aqi_labels = ['Good', 'Moderate', 'Unhealthy\nSensitive', 'Unhealthy', 'Very\nUnhealthy', 'Hazardous']

    for i in range(len(aqi_levels)):
        ax2.barh(0, aqi_levels[i] - (aqi_levels[i-1] if i > 0 else 0),
                 left=aqi_levels[i-1] if i > 0 else 0,
                 color=aqi_colors[i], alpha=0.7, label=aqi_labels[i])

    ax2.axvline(x=prediction, color='black', linestyle='--', linewidth=2, label=f'Current: {prediction:.1f}')
    ax2.set_xlim(0, 300)
    ax2.set_xlabel("PM2.5 (µg/m³)")
    ax2.set_title("AQI Scale Reference")
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_yticks([])
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Alerts
    if aqi_cat in ["Good", "Moderate"]:
        st.success(f"{city.title()} has relatively clean air. Keep promoting green habits 🌱")
    elif aqi_cat in ["Unhealthy for Sensitive Groups", "Unhealthy"]:
        st.warning(f"{city.title()} air quality is concerning. Sensitive groups should reduce outdoor exposure 😷")
    else:
        st.error(f"{city.title()} air quality is dangerous. Take urgent action 🚨")

# -------------------------------
# 📘 EDUCATION SECTION
# -------------------------------
with st.expander("📚 Learn About Air Quality Parameters"):
    st.markdown("""
### 🌬️ Understanding Key Pollutants:
- **PM2.5 & PM10:** Fine particles that penetrate lungs and bloodstream.  
- **NO, NO₂, NOx (🚗):** Traffic emissions — form smog and acid rain.  
- **SO₂ (🏭):** Emitted from coal-burning — causes respiratory irritation.  
- **O₃:** Ground-level ozone — harms lungs and crops.  
- **CO (🔥):** Reduces oxygen delivery to the body.  
- **VOCs (Benzene, Toluene, Xylene):** Industrial solvents with long-term health effects.

### 🌡️ AQI Categories:
| Category | Range | Health Impact |
|-----------|--------|---------------|
| ✅ Good | 0–50 | Minimal impact |
| 🙂 Moderate | 51–100 | Acceptable air quality |
| 😷 Sensitive | 101–150 | Affects heart/lung patients |
| ⚠️ Unhealthy | 151–200 | Health effects for all |
| ☠️ Very Unhealthy | 201–300 | Serious health warnings |
| 🚨 Hazardous | 301–500 | Emergency conditions |
""")
