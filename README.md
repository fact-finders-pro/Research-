# ML Model Deployment Pipeline with FastAPI

## Overview

This repository contains a modular machine learning pipeline covering:

- **Data Collection**: Load and verify data sources.
- **Data Preprocessing**: Clean, scale, and split data.
- **Model Inference**: Load a pre-trained model and serve predictions.
- **API Deployment**: FastAPI REST API for model inference.
- **Logging**: Track inputs and predictions for monitoring.
- **Dockerized Deployment**: Containerize the API for easy deployment.

---

## Folder Structure

project/
│
├── app/
│ ├── data_collection.py # Data loading and inspection
│ ├── data_preprocessing.py # Data cleaning, scaling, splitting
│ ├── model.py # Model loading and prediction function
│ ├── schema.py # Input data validation schema
│ ├── utils.py # Logging utilities
│ ├── main.py # FastAPI app entrypoint
│ └── models/
│ └── final_model.pkl # Pre-trained ML model
│
├── Dockerfile # Docker container definition
├── requirements.txt # Python dependencies
└── README.md # This documentation




