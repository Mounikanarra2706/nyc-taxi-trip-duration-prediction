from pyspark.sql import DataFrame
from pyspark.ml.evaluation import RegressionEvaluator


def evaluate_regression_model(predictions: DataFrame) -> dict:
    evaluator_rmse = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="rmse"
    )

    evaluator_mae = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="mae"
    )

    evaluator_r2 = RegressionEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="r2"
    )

    return {
        "RMSE": evaluator_rmse.evaluate(predictions),
        "MAE": evaluator_mae.evaluate(predictions),
        "R2": evaluator_r2.evaluate(predictions)
    }


def print_model_results(model_name: str, metrics: dict) -> None:
    print(f"\n{model_name}")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")