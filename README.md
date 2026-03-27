# NYC TAXI TRIP DURATION PREDICTION
# Overview
This project predicts NYC taxi trip duration using machine learning models. It implements a complete end-to-end pipeline including data processing, feature engineering, model training, and deployment for real-time inference.

## Problem Statement
Predict the duration of taxi trips in New York City using historical trip data and regression techniques.


## Tech Stack
- Python
- PySpark
- XGBoost (Spark)
- Scikit-learn
- FastAPI
- AWS EC2
- Pandas, NumPy

## Pipeline
Data Ingestion → Data Cleaning → Feature Engineering → Feature Vector → Model Training → Evaluation → API Deployment

# models used
- Linear Regression
- Decision Tree
- Random Forest
- XGBoost
- Tuned XGBoost

## Model Performance
Best Model: Tuned XGBoost
RMSE: 310.12
MAE: 202.35
R²: 0.7661

## Feature Engineering Highlights
- Time-based features (hour, weekday, month)
- Geospatial distance using Haversine formula
- Direction feature (bearing)
- Categorical encoding (vendor, flag)
- Custom flags (rush hour, weekend, night)

## API (FastAPI)
- Endpoints
- GET / → Health check
- POST /predict → Predict trip duration

## Example Request

{
  "vendor_id": "1",
  "pickup_datetime": "2016-05-10 08:15:00",
  "passenger_count": 1,
  "pickup_longitude": -73.985428,
  "pickup_latitude": 40.748817,
  "dropoff_longitude": -73.985130,
  "dropoff_latitude": 40.758896,
  "store_and_fwd_flag": "N"
}

## Example Response

{
  "predicted_trip_duration_seconds": 1201.93,
  "predicted_trip_duration_minutes": 20.03
}

## Deployment
- Deployed using FastAPI on AWS EC2
- Configured environment, dependencies, and server setup
- Handled memory and performance issues during deployment

## Project Sturcture
data/           → raw & processed data  
notebooks/      → EDA & experiments  
src/            → pipeline modules  
app/            → FastAPI application  
artifacts/      → saved models  