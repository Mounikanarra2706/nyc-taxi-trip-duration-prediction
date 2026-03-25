from pathlib import Path
from pyspark.sql import SparkSession
def create_spark_session(app_name:str="NYC Taxi Trip project")->SparkSession:
    """Create and return a SparkSession."""
    spark = SparkSession.builder.appName(app_name).master("local[*]").getOrCreate()
    return spark

def load_csv_data(spark: SparkSession, file_path: Path) -> SparkSession:
    """Load data from a CSV file into a Spark DataFrame."""
    file_path = str(file_path)
    df=spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)
    return df

def load_parquet_data(spark: SparkSession, file_path: Path) -> SparkSession:
    """Load data from a Parquet file into a Spark DataFrame."""
    file_path = str(file_path)
    df = spark.read.parquet(file_path)
    return df

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_data_path = project_root / "data" / "raw" / "train.csv"

    spark = create_spark_session()
    taxi_df = load_csv_data(spark, raw_data_path)

    taxi_df.printSchema()
    taxi_df.show(5, truncate=False)