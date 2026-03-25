from pyspark.sql import Row
from app.schemas import TripInput
from src.feature_engineering import feature_engineering_pipeline
from src.model_training import create_feature_vector


def predict_trip_duration(spark, model, vendor_indexer_model,flag_indexer_model,encoder_model,trip: TripInput) -> float:
    input_row = Row(
        vendor_id=trip.vendor_id,
        pickup_datetime=trip.pickup_datetime,
        passenger_count=trip.passenger_count,
        pickup_longitude=trip.pickup_longitude,
        pickup_latitude=trip.pickup_latitude,
        dropoff_longitude=trip.dropoff_longitude,
        dropoff_latitude=trip.dropoff_latitude,
        store_and_fwd_flag=trip.store_and_fwd_flag,
    )

    input_df = spark.createDataFrame([input_row])
    feature_df = feature_engineering_pipeline( input_df,vendor_indexer_model,flag_indexer_model,encoder_model)
    vector_df = create_feature_vector(feature_df)
    prediction_df = model.transform(vector_df)
    prediction = prediction_df.select("prediction").collect()[0][0]
    return float(prediction)