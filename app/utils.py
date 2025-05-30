import logging

# Setup logging
logging.basicConfig(filename="app/logs/predictions.log", level=logging.INFO)

def log_prediction(input_data: dict, prediction: str):
    logging.info(f"Input: {input_data}, Prediction: {prediction}")