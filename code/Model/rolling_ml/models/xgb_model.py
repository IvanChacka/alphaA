from imports import *
from env import MODEL_RANDOM_SEED, XGBOOST_N_JOBS, XGB_EARLY_STOPPING_ROUNDS, XGB_N_ESTIMATORS, XGB_OBJECTIVE
from rolling_ml.models.base_model import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, params=None):
        if XGBRegressor is None: raise ImportError("缺少xgboost，请执行：pip install xgboost")
        defaults = {"objective": XGB_OBJECTIVE, "tree_method": "hist", "n_estimators": XGB_N_ESTIMATORS,
                    "n_jobs": XGBOOST_N_JOBS, "random_state": MODEL_RANDOM_SEED}
        self.model = XGBRegressor(**{**defaults, **(params or {})})

    def fit(self, X_train, y_train, X_valid=None, y_valid=None):
        kwargs = {"verbose": False}
        if X_valid is not None:
            kwargs["eval_set"] = [(X_valid, y_valid)]
            self.model.set_params(early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS)
        self.model.fit(X_train, y_train, **kwargs); return self
    def predict(self, X): return np.asarray(self.model.predict(X), dtype=float)
    def get_feature_importance(self): return np.asarray(self.model.feature_importances_, dtype=float)
    @property
    def best_iteration(self): return getattr(self.model, "best_iteration", None)
