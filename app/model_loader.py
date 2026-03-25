from pathlib import Path

from pyspark.ml.feature import StringIndexerModel, OneHotEncoderModel
from xgboost.spark import SparkXGBRegressorModel
from src.data_ingestion import create_spark_session


def load_artifacts():
    spark = create_spark_session()
    project_root = Path(__file__).resolve().parent.parent
    artifacts_dir = project_root / "artifacts"
    model = SparkXGBRegressorModel.load(str(artifacts_dir / "tuned_xgboost_model"))
    vendor_indexer_model = StringIndexerModel.load(str(artifacts_dir / "vendor_indexer_model"))
    flag_indexer_model = StringIndexerModel.load(str(artifacts_dir / "flag_indexer_model"))
    encoder_model = OneHotEncoderModel.load(str(artifacts_dir / "onehot_encoder_model"))
    return spark, model, vendor_indexer_model, flag_indexer_model, encoder_model