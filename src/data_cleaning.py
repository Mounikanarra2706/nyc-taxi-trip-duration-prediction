from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

from src.data_ingestion import create_spark_session, load_csv_data

# dropping unnecessary columns 
def drop_unnecessary_columns(df: DataFrame) -> DataFrame:
    return df.drop("id")

# filtering out rows with invalid passenger count (0 or more than 6)
def filter_passenger_count(df: DataFrame) -> DataFrame:
    return df.filter((col("passenger_count") > 0) & (col("passenger_count") <= 6))

# filtering out trips with duration less than 1 minute or more than 2 hours (7200 seconds)
def filter_trip_duration(df: DataFrame) -> DataFrame:
    return df.filter((col("trip_duration") > 60) & (col("trip_duration") <= 7200))

# filtering out trips with pickup or dropoff coordinates outside of NYC bounding box
def filter_coordinates(df: DataFrame) -> DataFrame:
    return df.filter(
        (col("pickup_longitude").between(-74.05, -73.75)) &
        (col("pickup_latitude").between(40.63, 40.85)) &
        (col("dropoff_longitude").between(-74.05, -73.75)) &
        (col("dropoff_latitude").between(40.63, 40.85))
    )

# filtering out rows with null or invalid datetime values
def filter_valid_datetimes(df: DataFrame) -> DataFrame:
    return df.filter(
        col("pickup_datetime").isNotNull() &
        col("dropoff_datetime").isNotNull() &
        (col("pickup_datetime") < col("dropoff_datetime"))
    )
# removing duplicate rows

def remove_duplicates(df: DataFrame) -> DataFrame:
    return df.dropDuplicates()

# dropping columns that could cause data leakage (e.g., dropoff_datetime which can be used to calculate trip duration directly)
def drop_leakage_columns(df: DataFrame) -> DataFrame:
    return df.drop("dropoff_datetime")


def clean_data(df: DataFrame) -> DataFrame:
    df = drop_unnecessary_columns(df)
    df = filter_passenger_count(df)
    df = filter_trip_duration(df)
    df = filter_coordinates(df)
    df = filter_valid_datetimes(df)
    df = remove_duplicates(df)
    df = drop_leakage_columns(df)
    return df


if __name__ == "__main__":
    spark = create_spark_session()

    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "raw" / "train.csv"

    df = load_csv_data(spark, raw_path)
    clean_df = clean_data(df)

    print("Raw row count:", df.count())
    print("Clean row count:", clean_df.count())

    clean_df.show(5, truncate=False)