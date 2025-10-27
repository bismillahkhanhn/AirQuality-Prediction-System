import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load trained Linear Regression model
# -------------------------
lr_model = joblib.load("linear_regression_pm25.pkl")

# -------------------------
# Page setup
# -------------------------
st.set_page_config(page_title="Air Quality Prediction", layout="centered")
st.title("Air Quality Prediction System")



# -------------------------
# User Inputs
# -------------------------
city = st.text_input(" Enter City Name", "Delhi")

st.subheader(" Particulate Matter")
PM10 = st.number_input("PM10 (Coarse Particles µg/m³)", value=100.0)

st.subheader("🚗 Vehicular Pollutants")
NO = st.number_input("NO (Nitric Oxide µg/m³)", value=50.0)
NO2 = st.number_input("NO₂ (Nitrogen Dioxide µg/m³)", value=30.0)
NOx = st.number_input("NOx (Nitrogen Oxides µg/m³)", value=60.0)

st.subheader("🏭 Industrial & Chemical Pollutants")
NH3 = st.number_input("NH₃ (Ammonia µg/m³)", value=20.0)
CO = st.number_input("CO (Carbon Monoxide mg/m³)", value=2.0)
SO2 = st.number_input("SO₂ (Sulfur Dioxide µg/m³)", value=15.0)

st.subheader(" Atmospheric Gases")
O3 = st.number_input("O₃ (Ozone µg/m³)", value=40.0)

st.subheader(" Volatile Organic Compounds")
Benzene = st.number_input("Benzene (C₆H₆ µg/m³)", value=5.0)
Toluene = st.number_input("Toluene (C₇H₈ µg/m³)", value=4.0)
Xylene = st.number_input("Xylene (C₈H₁₀ µg/m³)", value=3.0)

# -------------------------
# Prediction
# -------------------------
if st.button(" Predict Air Quality", type="primary"):
    if not city.strip():
        st.error("❌ Please enter a valid city name!")
        st.stop()

    features = np.array([[PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene]])
    prediction = lr_model.predict(features)[0]

    # Convert PM2.5 to AQI Category with enhanced descriptions
    def pm25_to_aqi_category(pm):
        if pm <= 12: 
            return "Good", "green", "😊 Air quality is satisfactory with minimal health risk"
        elif pm <= 35.4: 
            return "Moderate", "yellow", "😐 Acceptable quality, but may affect sensitive individuals"
        elif pm <= 55.4: 
            return "Unhealthy for Sensitive Groups", "orange", "😷 Members of sensitive groups may experience health effects"
        elif pm <= 150.4: 
            return "Unhealthy", "red", "❗ Everyone may begin to experience health effects"
        elif pm <= 250.4: 
            return "Very Unhealthy", "purple", "⚠️ Health alert: serious risk to all population"
        else: 
            return "Hazardous", "maroon", "🚨 Emergency conditions: entire population affected"

    aqi_cat, color, desc = pm25_to_aqi_category(prediction)

    st.markdown(f"### Predicted PM2.5: **{prediction:.2f} µg/m³**")
    st.markdown(f"<h3 style='color:{color};'>AQI Category: {aqi_cat}</h3>", unsafe_allow_html=True)
    st.markdown(f"**{desc}**")

    # -------------------------
    # AQI Scale visualization
    # -------------------------
    fig, ax2 = plt.subplots(figsize=(6, 2))  # Adjusted height
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

    # -------------------------
    # Alert Message
    # -------------------------
    if aqi_cat in ["Good", "Moderate"]:
        st.success(f"{city.title()} has clean air quality. Keep promoting eco-friendly habits 🌱")
    elif aqi_cat in ["Unhealthy for Sensitive Groups", "Unhealthy"]:
        st.warning(f"{city.title()} has high pollution levels. Sensitive groups should reduce outdoor exposure 😷")
    else:
        st.error(f"{city.title()} has dangerously high pollution levels. Implement emergency measures and public health advisories 🚨")

# -------------------------
# Learn Section
# -------------------------
with st.expander("📚 Learn About Air Quality Parameters"):
    st.markdown("""
### Understanding Key Pollutants:
- **PM2.5 & PM10:** Tiny particles that enter the lungs and bloodstream  
- **NO, NO₂, NOx (🚗):** From vehicles and combustion — major smog contributors  
- **SO₂ (🏭):** From burning fuels containing sulfur — causes acid rain  
- **O₃:** Ground-level ozone — harms lungs and crops  
- **CO (🔥):** Reduces oxygen delivery to body tissues  
- **VOCs (Benzene, Toluene, Xylene):** Industrial chemicals linked to long-term health effects  

### AQI Categories:
| Category | AQI Range | Health Impact |
|-----------|------------|----------------|
| Good | 0–50 | Minimal impact |
| Moderate | 51–100 | Acceptable for most |
| Sensitive | 101–150 | Affects heart/lung patients |
| Unhealthy | 151–200 | Everyone may feel effects |
| Very Unhealthy | 201–300 | Health warnings issued |
| Hazardous | 301–500 | Emergency conditions |
""")

# -------------------------
# Pollutant Range Table (Hidden)
# -------------------------
with st.expander("📏 View Pollutant Reference Ranges"):
    st.markdown("""
| Pollutant | Typical Safe Range | Unit |
|------------|--------------------|------|
| PM10 | 0 – 600 | µg/m³ |
| NO | 0 – 500 | µg/m³ |
| NO₂ | 0 – 500 | µg/m³ |
| NOx | 0 – 500 | µg/m³ |
| NH₃ | 0 – 200 | µg/m³ |
| CO | 0.0 – 50.0 | mg/m³ |
| SO₂ | 0 – 300 | µg/m³ |
| O₃ | 0 – 300 | µg/m³ |
| Benzene | 0.0 – 50.0 | µg/m³ |
| Toluene | 0.0 – 50.0 | µg/m³ |
| Xylene | 0.0 – 50.0 | µg/m³ |
""")
# -------------------------
# Custom Page Styling (Eco Theme)
# -------------------------

   # -------------------------
# Simple Green Theme (Beginner Friendly)
# -------------------------
st.markdown("""
    <style>
    /* White background */
    [data-testid="stAppViewContainer"] {
        background-color: white;
        color: #1b5e20;
    }

    /* Green titles and headings */
    h1, h2, h3, h4 {
        color: #1b5e20 !important;
    }

    /* Green button */
    div.stButton > button:first-child {
        background-color: #2e7d32;
        color: white;
        border-radius: 6px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #43a047;
    }
    </style>
""", unsafe_allow_html=True)
