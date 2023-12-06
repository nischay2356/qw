import logging
from zenml import step
import pandas as pd

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    config: ModelNameConfig,
) -> RegressorMixin:
    try:
        model = None
        if config.model_name == 'LinearRegression':
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("MOdel{} not supported".format(config.model_name))
    except Exception as e:
        logging.error("error in traingin model: {}".format(e))
        raise e
