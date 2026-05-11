from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib
import os

# =========================
# DEBUG FILES
# =========================

print("Current Files:")
print(os.listdir())

# =========================
# FASTAPI APP
# =========================

app = FastAPI()

# =========================
# LOAD MODEL + SCALER
# =========================

model = None
scaler = None

try:

    print("Loading Random Forest model...")

    model = joblib.load("aqi_rf_model.pkl")

    print("Random Forest Model loaded successfully")

except Exception as e:

    print("MODEL ERROR:", e)

try:

    print("Loading scaler...")

    scaler = joblib.load("scaler.pkl")

    print("Scaler loaded successfully")

except Exception as e:

    print("SCALER ERROR:", e)

# =========================
# LATEST DATA STORAGE
# =========================

latest_data = {}

# =========================
# INPUT MODEL
# =========================

class SensorData(BaseModel):

    temperature: float
    humidity: float
    dust: float
    mq135: float
    mq2: float

# =========================
# HOME ROUTE
# =========================

@app.get("/")
def home():

    return {

        "message": "AQI Random Forest Prediction API Running"
    }

# =========================
# LATEST DATA ROUTE
# =========================

@app.get("/latest")
def latest():

    return latest_data

# =========================
# PREDICT ROUTE
# =========================

@app.post("/predict")
def predict(data: SensorData):

    global latest_data

    try:

        # =========================
        # CHECK MODEL
        # =========================

        if model is None or scaler is None:

            return {

                "status": "error",

                "message": "Model or scaler failed to load"
            }

        # =========================
        # CREATE FEATURE DATAFRAME
        # =========================

        features = pd.DataFrame([{

            "pm": data.dust,

            "MQ135": data.mq135,

            "co": data.mq2,

            "temperature": data.temperature,

            "humidity": data.humidity
        }])

        # =========================
        # SCALE FEATURES
        # =========================

        scaled_features = scaler.transform(features)

        # =========================
        # PREDICT AQI
        # =========================

        predicted_aqi = model.predict(
            scaled_features
        )[0]

        # =========================
        # FUTURE AQI ESTIMATION
        # =========================

        future_aqi = predicted_aqi * 1.05

        current_aqi = predicted_aqi

        # =========================
        # AQI CATEGORY
        # =========================

        if current_aqi <= 50:

            category = "Good"

        elif current_aqi <= 100:

            category = "Satisfactory"

        elif current_aqi <= 200:

            category = "Moderate"

        elif current_aqi <= 300:

            category = "Poor"

        elif current_aqi <= 400:

            category = "Very Poor"

        else:

            category = "Severe"

        # =========================
        # STORE LATEST DATA
        # =========================

        latest_data = {

            "status": "success",

            "temperature": data.temperature,

            "humidity": data.humidity,

            "dust": data.dust,

            "mq135": data.mq135,

            "mq2": data.mq2,

            "current_aqi": round(current_aqi, 2),

            "future_aqi": round(future_aqi, 2),

            "category": category
        }

        # =========================
        # RETURN RESPONSE
        # =========================

        return latest_data

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }