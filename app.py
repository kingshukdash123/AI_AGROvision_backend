import os
import pickle
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Crop Yield Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model with fallback
model = None
if os.path.exists("Making_model/models/yield_model.joblib"):
    model = joblib.load("Making_model/models/yield_model.joblib")
# elif os.path.exists("crop_yield_model.pkl"):
#     try:
#         model = joblib.load("crop_yield_model.pkl")
#     except Exception:
#         with open("crop_yield_model.pkl", "rb") as f:
#             model = pickle.load(f)
else:
    raise FileNotFoundError("Model file not found. Expected 'yield_model.joblib'")

# Initialize FastAPI


# Input schema
class CropInput(BaseModel):
    crop: str
    season: str
    state: str
    area: float
    annual_rainfall: str

# Root endpoint
@app.get("/")
def home():
    return {"message": "Welcome to Crop Yield Prediction API"}

# Prediction endpoint


@app.post("/predict")
def predict(data: CropInput):
    try:
        # Create DataFrame with correct column names
        input_df = pd.DataFrame([{
            "Crop": data.crop,
            "Season": data.season,
            "State": data.state,
            "Area": data.area,
            "Annual_Rainfall": data.annual_rainfall
        }])

        # Predict production using model
        production = model.predict(input_df)[0]
        yield_value = production / data.area if data.area != 0 else 0

        return {
            "Crop": data.crop,
            "Season": data.season,
            "State": data.state,
            "Area (ha)": data.area,
            "Annual Rainfall": data.annual_rainfall,
            "Predicted Production (tons)": round(float(production), 2),
            "Predicted Yield (tons/ha)": round(float(yield_value), 2)
        }

    except Exception as e:
        return {"error": str(e)}
