from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, hour, month, dayofweek, when,
    radians, sin, cos, sqrt, asin, atan2
)
from pyspark.ml.feature import StringIndexer, OneHotEncoder


def add_time_features(df: DataFrame) -> DataFrame:
    df = df.withColumn("pickup_hour", hour(col("pickup_datetime")))
    df = df.withColumn("pickup_month", month(col("pickup_datetime")))
    df = df.withColumn("pickup_weekday", dayofweek(col("pickup_datetime")))
    return df


def add_is_weekend(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "is_weekend",
        when(col("pickup_weekday").isin(1, 7), 1).otherwise(0)
    )


def add_is_rush_hour(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "is_rush_hour",
        when(
            ((col("pickup_hour") >= 7) & (col("pickup_hour") <= 9)) |
            ((col("pickup_hour") >= 16) & (col("pickup_hour") <= 19)),
            1
        ).otherwise(0)
    )


def add_is_night(df: DataFrame) -> DataFrame:
    return df.withColumn(
        "is_night",
        when((col("pickup_hour") >= 22) | (col("pickup_hour") <= 5), 1).otherwise(0)
    )


def add_trip_distance(df: DataFrame) -> DataFrame:
    df = df.withColumn("pickup_lat_rad", radians(col("pickup_latitude")))
    df = df.withColumn("pickup_lon_rad", radians(col("pickup_longitude")))
    df = df.withColumn("dropoff_lat_rad", radians(col("dropoff_latitude")))
    df = df.withColumn("dropoff_lon_rad", radians(col("dropoff_longitude")))
    df = df.withColumn("lat_diff", col("dropoff_lat_rad") - col("pickup_lat_rad"))
    df = df.withColumn("lon_diff", col("dropoff_lon_rad") - col("pickup_lon_rad"))
    df = df.withColumn("a", sin(col("lat_diff") / 2) ** 2 +cos(col("pickup_lat_rad")) 
                       * cos(col("dropoff_lat_rad")) *sin(col("lon_diff") / 2) ** 2)
    df = df.withColumn("trip_distance_km", 2 * 6371 * asin(sqrt(col("a"))))

    return df.drop("pickup_lat_rad", "pickup_lon_rad","dropoff_lat_rad", "dropoff_lon_rad", "lat_diff", "lon_diff", "a")


def add_direction_feature(df: DataFrame) -> DataFrame:
    return (df.withColumn("latitude_difference", col("dropoff_latitude") - col("pickup_latitude")).withColumn("longitude_difference", col("dropoff_longitude") - col("pickup_longitude"))
        .withColumn("direction", atan2(col("longitude_difference"), col("latitude_difference")))
    )


def fit_categorical_transformers(df: DataFrame):
    vendor_indexer = StringIndexer(inputCol="vendor_id",outputCol="vendor_id_index",handleInvalid="keep")
    flag_indexer = StringIndexer(inputCol="store_and_fwd_flag",outputCol="store_and_fwd_flag_index",handleInvalid="keep")

    vendor_indexer_model = vendor_indexer.fit(df)
    flag_indexer_model = flag_indexer.fit(df)

    indexed_df = vendor_indexer_model.transform(df)
    indexed_df = flag_indexer_model.transform(indexed_df)

    encoder = OneHotEncoder(inputCols=["vendor_id_index", "store_and_fwd_flag_index"],outputCols=["vendor_id_encoded", "store_and_fwd_flag_encoded"])

    encoder_model = encoder.fit(indexed_df)

    return vendor_indexer_model, flag_indexer_model, encoder_model


def transform_categorical_columns(df: DataFrame,vendor_indexer_model,flag_indexer_model,encoder_model) -> DataFrame:
    df = vendor_indexer_model.transform(df)
    df = flag_indexer_model.transform(df)
    df = encoder_model.transform(df)

    return df.drop("vendor_id_index", "store_and_fwd_flag_index")


def feature_engineering_pipeline(df: DataFrame,vendor_indexer_model=None,flag_indexer_model=None,encoder_model=None) -> DataFrame:
    df = add_time_features(df)
    df = add_is_weekend(df)
    df = add_is_rush_hour(df)
    df = add_is_night(df)
    df = add_trip_distance(df)
    df = add_direction_feature(df)

    if vendor_indexer_model and flag_indexer_model and encoder_model:
        df = transform_categorical_columns(df,vendor_indexer_model,flag_indexer_model,encoder_model)
    else:
        raise ValueError("Fitted categorical transformer models are required.")

    return df

if __name__ == "__main__":
    from pathlib import Path
    from src.data_ingestion import create_spark_session, load_csv_data
    from src.data_cleaning import clean_data

    spark = create_spark_session()

    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw" / "train.csv"

    df = load_csv_data(spark, raw_path)
    clean_df = clean_data(df)
    feature_df = feature_engineering_pipeline(clean_df)
    feature_df.show(5, truncate=False)
    feature_df.columns
