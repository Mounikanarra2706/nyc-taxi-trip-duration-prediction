from pyspark.sql import DataFrame
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, DecisionTreeRegressor, RandomForestRegressor
from pathlib import Path
from src.data_ingestion import create_spark_session, load_csv_data, load_parquet_data
from src.data_cleaning import clean_data
from src.feature_engineering import feature_engineering_pipeline, fit_categorical_transformers
from src.evaluation import evaluate_regression_model, print_model_results
from src.xgboost_training import (
    train_xgboost,
    tune_xgboost,
    evaluate_xgboost_model,
    print_xgboost_results,
)

#by using the vector assembler we can combine all the features into a single vector column which is required for Spark ML models. 
# This function takes a DataFrame as input and returns a new DataFrame with a "features" column and a "label" column (which is the trip duration).
def create_feature_vector(df: DataFrame) -> DataFrame:
    feature_cols = [
        "passenger_count",
        "pickup_hour",
        "pickup_month",
        "pickup_weekday",
        "is_weekend",
        "is_rush_hour",
        "is_night",
        "trip_distance_km",
        "latitude_difference",
        "longitude_difference",
        "direction",
        "vendor_id_encoded",
        "store_and_fwd_flag_encoded",
    ]

    assembler = VectorAssembler(inputCols=feature_cols,outputCol="features")
    df = assembler.transform(df)
    return df
def assemble_features(df: DataFrame) -> DataFrame:
    df = create_feature_vector(df)
    df = df.select("features", "trip_duration")
    return df.withColumnRenamed("trip_duration", "label")

# The randomSplit function is used to split the DataFrame into two parts: 80% for training and 20% for testing.
#  The seed parameter ensures that the split is reproducible.
def split_data(df: DataFrame):
    return df.randomSplit([0.8, 0.2], seed=42)
# This function trains three different regression models (Linear Regression, Decision Tree, and Random Forest) on the training data and evaluates their performance on the test data.

def train_spark_models(train_df: DataFrame, test_df: DataFrame):
    results = {}
    lr = LinearRegression(featuresCol="features", labelCol="label")
    lr_model = lr.fit(train_df)
    lr_pred = lr_model.transform(test_df)
    results["Linear Regression"] = evaluate_regression_model(lr_pred)

    dt = DecisionTreeRegressor(featuresCol="features", labelCol="label")
    dt_model = dt.fit(train_df)
    dt_pred = dt_model.transform(test_df)
    results["Decision Tree"] = evaluate_regression_model(dt_pred)

    rf = RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=50)
    rf_model = rf.fit(train_df)
    rf_pred = rf_model.transform(test_df)
    results["Random Forest"] = evaluate_regression_model(rf_pred)

    return results, rf_pred

# This function converts the Spark DataFrames for training and testing into NumPy arrays that can be used with XGBoost.
def convert_spark_to_numpy(train_df: DataFrame, test_df: DataFrame):
    train_pd = train_df.select("features", "label").toPandas()
    test_pd = test_df.select("features", "label").toPandas()

    train_X = np.array(train_pd["features"].apply(lambda x: x.toArray()).tolist())
    train_y = train_pd["label"].values

    test_X = np.array(test_pd["features"].apply(lambda x: x.toArray()).tolist())
    test_y = test_pd["label"].values

    return train_X, train_y, test_X, test_y


def main():

    spark = create_spark_session()

    project_root = Path(__file__).resolve().parent.parent
    raw_data_path = project_root / "data" / "raw" / "train.csv"

    # 1. Load raw data
    df = load_csv_data(spark, raw_data_path)

    # 2. Clean data
    clean_df = clean_data(df)

    # 3. Fit categorical transformers
    vendor_indexer_model, flag_indexer_model, encoder_model = fit_categorical_transformers(clean_df)

    # 4. Save preprocessing models
    artifacts_dir = project_root / "artifacts"
    vendor_indexer_model.write().overwrite().save(str(artifacts_dir / "vendor_indexer_model"))
    flag_indexer_model.write().overwrite().save(str(artifacts_dir / "flag_indexer_model"))
    encoder_model.write().overwrite().save(str(artifacts_dir / "onehot_encoder_model"))

    # 5. Feature engineering using fitted transformers
    feature_df = feature_engineering_pipeline(clean_df,vendor_indexer_model,flag_indexer_model,encoder_model)
    # 6. Assemble features
    final_df = assemble_features(feature_df)
    # 7. Train-test split
    train_df, test_df = split_data(final_df)
    # 8. Train Spark models
    spark_results, _ = train_spark_models(train_df, test_df)
    print("\nSpark Model Results")
    for model_name, metrics in spark_results.items():
        print_model_results(model_name, metrics)
    # 9. Baseline XGBoost
    xgb_model = train_xgboost(train_df)
    xgb_pred = xgb_model.transform(test_df)
    xgb_metrics = evaluate_xgboost_model(xgb_pred)
    print_xgboost_results("XGBoost", xgb_metrics)
    # 10. Tuned XGBoost
    cv_model = tune_xgboost(train_df)
    best_xgb_model = cv_model.bestModel
    tuned_xgb_pred = best_xgb_model.transform(test_df)
    tuned_xgb_metrics = evaluate_xgboost_model(tuned_xgb_pred)
    print_xgboost_results("Tuned XGBoost", tuned_xgb_metrics)

    print("\nBest XGBoost Parameters:")
    print(best_xgb_model.extractParamMap())

    # 11. Save model
    model_save_path = project_root / "artifacts" / "tuned_xgboost_model"
    best_xgb_model.write().overwrite().save(str(model_save_path))

    print(f"Best XGBoost model saved to: {model_save_path}")

   

    


if __name__ == "__main__":
    main()