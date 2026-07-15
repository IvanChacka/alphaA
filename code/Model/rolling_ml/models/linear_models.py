from imports import *
from env import MODEL_N_JOBS, MODEL_RANDOM_SEED
from rolling_ml.models.base_model import BaseModel


class LinearModel(BaseModel):
    def __init__(self, name: str = "linear_regression", params: dict | None = None):
        params = params or {}
        factories = {"linear_regression": LinearRegression, "ridge": Ridge, "elasticnet": ElasticNet}
        if name not in factories:
            raise ValueError(f"未知线性模型：{name}")
        if name == "elasticnet":
            params = {"max_iter": 5000, "random_state": MODEL_RANDOM_SEED, **params}
        self.name, self.model = name, factories[name](**params)

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        # Ridge/LinearRegression底层BLAS使用统一线程数；ElasticNet实现本身可能无法完全并行。
        with threadpool_limits(limits=MODEL_N_JOBS):
            self.model.fit(X_train, y_train)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)

    def get_feature_importance(self) -> np.ndarray:
        return np.asarray(getattr(self.model, "coef_", []), dtype=float)
