import joblib
import pandas as pd

# Load trained model
model = joblib.load("app/models/final_model.pkl")

def predict(input_data: dict):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    return prediction