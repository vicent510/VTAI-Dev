# Dependencies Imports

# Local Imports
from utils.basics import _log

def train_prediction_model(train_prediction_model_config: dict):
    train_source_path = train_prediction_model_config.get("train_source_path")
    val_source_path = train_prediction_model_config.get("val_source_path")
    test_source_path = train_prediction_model_config.get("test_source_path")

    