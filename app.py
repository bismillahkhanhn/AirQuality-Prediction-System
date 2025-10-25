

# Import Libraries
import ipywidgets as widgets
from IPython.display import display
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load trained Linear Regression model
lr_model = joblib.load("linear_regression_pm25.pkl")

# --- UI Components --- #
st_title = widgets.HTML("<h2 style='color:green;'>🌬️ PM2.5 Prediction App (Linear Regression)</h2>")
display(st_title)

# Slider inputs for pollutants
PM10 = widgets.IntSlider(min=0, max=600, step=1, value=100, description='PM10')
NO = widgets.IntSlider(min=0, max=500, step=1, value=50, description='NO')
NO2 = widgets.IntSlider(min=0, max=500, step=1, value=30, description='NO2')
NOx = widgets.IntSlider(min=0, max=500, step=1, value=60, description='NOx')
NH3 = widgets.IntSlider(min=0, max=200, step=1, value=20, description='NH3')
CO = widgets.FloatSlider(min=0, max=50, step=0.1, value=2, description='CO')
SO2 = widgets.IntSlider(min=0, max=300, step=1, value=15, description='SO2')
O3 = widgets.IntSlider(min=0, max=300, step=1, value=40, description='O3')
Benzene = widgets.FloatSlider(min=0, max=50, step=0.1, value=5, description='Benzene')
Toluene = widgets.FloatSlider(min=0, max=50, step=0.1, value=4, description='Toluene')
Xylene = widgets.FloatSlider(min=0, max=50, step=0.1, value=3, description='Xylene')

# Button to predict
predict_button = widgets.Button(description="Predict PM2.5", button_style='success', tooltip="Click to predict")

# Output widget
output = widgets.Output()

# --- Layout --- #
ui = widgets.VBox([PM10, NO, NO2, NOx, NH3, CO, SO2, O3, Benzene, Toluene, Xylene, predict_button, output])
display(ui)

# --- Prediction Function --- #
def pm25_to_aqi_category(pm):
    if pm <= 12: return "Good"
    elif pm <= 35.4: return "Moderate"
    elif pm <= 55.4: return "Unhealthy (Sensitive)"
    elif pm <= 150.4: return "Unhealthy"
    elif pm <= 250.4: return "Very Unhealthy"
    else: return "Hazardous"

def on_predict_clicked(b):
    with output:
        output.clear_output()
        features = np.array([[PM10.value, NO.value, NO2.value, NOx.value, NH3.value, CO.value,
                              SO2.value, O3.value, Benzene.value, Toluene.value, Xylene.value]])
        prediction = lr_model.predict(features)[0]
        aqi_cat = pm25_to_aqi_category(prediction)
        print(f"✅ Predicted PM2.5: {prediction:.2f}")
        print(f"🌡️ AQI Category: {aqi_cat}")
        
        # Optional: Visualize prediction
        plt.figure(figsize=(6,4))
        plt.bar(['PM2.5'], [prediction], color='red')
        plt.ylim(0, max(prediction+50, 200))
        plt.ylabel("PM2.5 Value")
        plt.title("Predicted PM2.5")
        plt.show()

predict_button.on_click(on_predict_clicked)
