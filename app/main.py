from fastapi import FastAPI
from app.schema import ModelInput
from app.model import predict
from app.utils import log_prediction

app = FastAPI(title="ML Model Inference API")

@app.post("/predict")
def get_prediction(data: ModelInput):
    input_dict = data.dict()
    prediction = predict(input_dict)
    log_prediction(input_dict, prediction)
    return {"prediction": prediction}