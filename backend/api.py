from fastapi import FastAPI
import pickle
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",  # frontend URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("diabetes_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    if prediction == 0:
        return {
            "result": "No Diabetes ✅",
            "tips": [
                "Maintain a balanced diet 🥗",
                "Exercise regularly 💪",
                "Keep a healthy BMI"
            ]
        }
    else:
        return {
            "result": "Diabetes Detected ⚠️",
            "medicines": [
                "Metformin (first-line treatment)",
                "Sulfonylureas (e.g., Glipizide, Glimepiride)",
                "Insulin therapy (in some cases)"
            ],
            "tips": [
                "Follow a low-sugar, high-fiber diet 🍎",
                "Exercise at least 30 minutes daily 🏃",
                "Monitor blood sugar regularly",
                "Consult a doctor for personalized treatment"
            ]
        }
