from imports import *
from rolling_ml.models.linear_models import LinearModel
from rolling_ml.models.xgb_model import XGBoostModel


def create_model(name: str, params=None):
    return XGBoostModel(params) if name == "xgboost" else LinearModel(name, params)

