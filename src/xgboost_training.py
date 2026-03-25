from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from xgboost.spark import SparkXGBRegressor


def train_xgboost(train_df: DataFrame) -> SparkXGBRegressor:
    """
    Train a baseline Spark XGBoost model.
    """
    xgb = SparkXGBRegressor(features_col="features",label_col="label")
    model = xgb.fit(train_df)
    return model


def tune_xgboost(train_df: DataFrame) -> CrossValidator:
    """
    Tune Spark XGBoost using ParamGridBuilder + CrossValidator.
    """
    xgb = SparkXGBRegressor(features_col="features",label_col="label",prediction_col="prediction", num_workers=2,objective="reg:squarederror",)

    param_grid = (
        ParamGridBuilder()
        .addGrid(xgb.max_depth, [3,5,7])
        .addGrid(xgb.learning_rate, [0.05, 0.1])
        .addGrid(xgb.n_estimators, [100, 200, 300])
        .build()
    )

    evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse"
    )

    cross_validator = CrossValidator(estimator=xgb,estimatorParamMaps=param_grid,evaluator=evaluator,numFolds=2 )

    cv_model = cross_validator.fit(train_df)
    return cv_model


def evaluate_xgboost_model(predictions: DataFrame) -> dict:
    """
    Evaluate Spark prediction DataFrame.
    """
    evaluator_rmse = RegressionEvaluator(labelCol="label",predictionCol="prediction",metricName="rmse")

    evaluator_mae = RegressionEvaluator(labelCol="label",predictionCol="prediction", metricName="mae")

    evaluator_r2 = RegressionEvaluator(labelCol="label",predictionCol="prediction", metricName="r2" )

    return {
        "RMSE": evaluator_rmse.evaluate(predictions),
        "MAE": evaluator_mae.evaluate(predictions),
        "R2": evaluator_r2.evaluate(predictions)
    }


def print_xgboost_results(model_name: str, metrics: dict) -> None:
    print(f"\n{model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")