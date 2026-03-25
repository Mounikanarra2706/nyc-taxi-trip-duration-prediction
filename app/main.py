from fastapi import FastAPI, HTTPException
from app.schemas import TripInput
from app.model_loader import load_artifacts
from app.predict import predict_trip_duration

app = FastAPI(title="NYC Taxi Trip Duration Prediction API")

spark, model, vendor_indexer_model, flag_indexer_model, encoder_model = load_artifacts()
@app.get("/")
def home():
    return {"message": "NYC Taxi Trip Duration API is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(trip: TripInput):
    try:
        predicted_duration = predict_trip_duration(spark,model,vendor_indexer_model,
            flag_indexer_model,encoder_model,trip)
        return {
            "predicted_trip_duration_seconds": round(predicted_duration, 2),
            "predicted_trip_duration_minutes": round(predicted_duration / 60, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))