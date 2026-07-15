from imports import *
from env import (MODEL_N_JOBS, STYLE_XGB_EARLY_STOPPING_ROUNDS,
                 STYLE_XGB_MAX_ESTIMATORS, STYLE_XGB_VALIDATION_EMBARGO_DAYS)


class StyleTimer:
    def __init__(self, name: str, styles: list[str], params: dict | None = None):
        self.name, self.styles, self.params = name, styles, params or {}
        self.model: Any = None

    @property
    def feature_columns(self):
        return [f"{s}_{suffix}" for s in self.styles
                for suffix in ("momentum_5", "momentum_20", "volatility_5")] + ["benchmark_volatility_20"]

    def fit(self, data: pd.DataFrame):
        if self.name in {"equal_weight", "momentum_rule"}: return self
        x = data[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        if self.name == "multinomial_logistic":
            self.model = LogisticRegression(C=self.params.get("C", 1.0), max_iter=1000,
                                            random_state=42)
            self.model.fit(x, data.best_style, sample_weight=data.sample_weight)
        elif self.name == "xgboost_classifier":
            if XGBClassifier is None: raise ImportError("xgboost未安装")
            mapping = {s: i for i, s in enumerate(self.styles)}
            self.model = XGBClassifier(n_estimators=self.params.get("n_estimators", 200),
                max_depth=self.params.get("max_depth", 3), learning_rate=self.params.get("learning_rate", .03),
                subsample=.8, colsample_bytree=.8, n_jobs=MODEL_N_JOBS, random_state=42, eval_metric="mlogloss")
            self.model.fit(x, data.best_style.map(mapping), sample_weight=data.sample_weight)
        elif self.name == "xgboost_regressors":
            if XGBRegressor is None: raise ImportError("xgboost未安装")
            self.model = {}
            for style in self.styles:
                reg = XGBRegressor(n_estimators=self.params.get("n_estimators", 200), max_depth=3,
                    learning_rate=.03, subsample=.8, colsample_bytree=.8, n_jobs=MODEL_N_JOBS, random_state=42)
                reg.fit(x, data[f"target_{style}"], sample_weight=data.sample_weight)
                self.model[style] = reg
        return self

    def predict_weights(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.name == "equal_weight":
            values = np.full((len(data), len(self.styles)), 1 / len(self.styles))
        elif self.name == "momentum_rule":
            raw = data[[f"{s}_momentum_20" for s in self.styles]].fillna(0).to_numpy()
            raw -= raw.max(axis=1, keepdims=True)
            values = np.exp(raw * 20); values /= values.sum(axis=1, keepdims=True)
        else:
            x = data[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
            if self.name == "multinomial_logistic":
                values = np.zeros((len(data), len(self.styles)))
                probability = self.model.predict_proba(x)
                for j, label in enumerate(self.model.classes_): values[:, self.styles.index(label)] = probability[:, j]
            elif self.name == "xgboost_classifier":
                values = self.model.predict_proba(x)
            else:
                raw = np.column_stack([self.model[s].predict(x) for s in self.styles])
                raw -= raw.max(axis=1, keepdims=True)
                values = np.exp(raw * 20); values /= values.sum(axis=1, keepdims=True)
        out = pd.DataFrame(values, columns=self.styles)
        out.insert(0, "decision_date", data.decision_date.to_numpy())
        return out


class DualHorizonXGBoost:
    """用一个共享XGB混合所有风格特征，预测每个风格的未来收益。

    训练表按“决策日×目标风格”展开：每行同时包含所有风格动量、
    波动率、市场状态与目标风格标识。这样各风格预测值来自同一模型、
    处于同一收益尺度，可直接做跨风格排序。
    """

    def __init__(self, styles: list[str], horizon: int, params: dict | None = None):
        if XGBRegressor is None:
            raise ImportError("xgboost未安装")
        self.styles, self.horizon, self.params = styles, horizon, params or {}
        self.model: Any = None
        self.validation_rank_ic = np.nan
        self.best_iterations: list[int] = []
        self.validation_start_: pd.Timestamp | None = None
        self.fit_end_: pd.Timestamp | None = None
        self.embargo_days_ = STYLE_XGB_VALIDATION_EMBARGO_DAYS

    @property
    def feature_columns(self) -> list[str]:
        return [f"{style}_{suffix}" for style in self.styles
                for suffix in ("momentum_5", "momentum_20", "volatility_5")] + [
                    "benchmark_volatility_20", "benchmark_return_20"]

    @property
    def style_indicator_columns(self) -> list[str]:
        return [f"target_is_{style}" for style in self.styles]

    @property
    def target_style_feature_columns(self) -> list[str]:
        return ["target_momentum_5", "target_momentum_20", "target_volatility_5"]

    @property
    def model_feature_columns(self) -> list[str]:
        return self.feature_columns + self.target_style_feature_columns + self.style_indicator_columns

    def _long_matrix(self, data: pd.DataFrame, include_target: bool = False
                     ) -> tuple[pd.DataFrame, np.ndarray | None, np.ndarray | None]:
        """将每个日期展开为多个目标风格样本，供一个共享XGB联合学习。"""
        base = data[self.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
        style_count = len(self.styles)
        repeated = np.repeat(base.to_numpy(dtype=np.float32, copy=False), style_count, axis=0)
        style_ids = np.tile(np.arange(style_count), len(data))
        own_features = np.column_stack([
            data[[f"{style}_{suffix}" for style in self.styles]].replace(
                [np.inf, -np.inf], np.nan).fillna(0).to_numpy(dtype=np.float32, copy=False).reshape(-1)
            for suffix in ("momentum_5", "momentum_20", "volatility_5")])
        indicators = np.eye(style_count, dtype=np.float32)[style_ids]
        x = pd.DataFrame(np.column_stack([repeated, own_features, indicators]),
                         columns=self.model_feature_columns)
        if not include_target:
            return x, None, None
        targets = data[[f"target_{style}_{self.horizon}" for style in self.styles]].to_numpy(
            dtype=np.float32, copy=False).reshape(-1)
        weights = np.repeat(data.sample_weight.to_numpy(dtype=np.float32, copy=False), style_count)
        return x, targets, weights

    @staticmethod
    def _purged_time_split(data: pd.DataFrame, embargo_days: int = STYLE_XGB_VALIDATION_EMBARGO_DAYS
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """最后20%作验证集，并在拟合集和验证集之间留出固定交易日隔离区。"""
        ordered = data.sort_values("decision_date").reset_index(drop=True)
        dates = pd.DatetimeIndex(ordered.decision_date.drop_duplicates().sort_values())
        if len(dates) <= embargo_days + 2:
            raise ValueError(f"择时XGB样本不足，无法留出{embargo_days}个交易日隔离期")
        valid_position = max(embargo_days + 1, min(len(dates) - 1, int(len(dates) * .80)))
        fit_end_position = valid_position - embargo_days
        fit_dates, valid_dates = dates[:fit_end_position], dates[valid_position:]
        fit_data = ordered[ordered.decision_date.isin(fit_dates)].copy()
        valid_data = ordered[ordered.decision_date.isin(valid_dates)].copy()
        if fit_data.empty or valid_data.empty:
            raise ValueError("择时XGB隔离后拟合集或验证集为空")
        return fit_data, valid_data

    def fit(self, data: pd.DataFrame, selected_trees: int | None = None,
            validation_rank_ic: float | None = None,
            selection_fit_end: pd.Timestamp | None = None,
            selection_validation_start: pd.Timestamp | None = None):
        """拟合共享模型；仅首次选择树数时建立带隔离期的验证集。

        后续季度传入固定 ``selected_trees``，直接使用截止当季前的全部已实现
        标签重训，不再重复切走验证集或隔离区。
        """
        ordered = data.sort_values("decision_date").reset_index(drop=True)
        x_all, y_all, weight_all = self._long_matrix(ordered, include_target=True)
        common = {"max_depth": self.params.get("max_depth", 2),
            "learning_rate": self.params.get("learning_rate", .03),
            "min_child_weight": self.params.get("min_child_weight", 1.0),
            "gamma": self.params.get("gamma", 0.0),
            "reg_alpha": self.params.get("reg_alpha", 0.0),
            "reg_lambda": self.params.get("reg_lambda", 1.0),
            "subsample": .75, "colsample_bytree": .70, "n_jobs": MODEL_N_JOBS,
            "random_state": 42, "objective": "reg:squarederror", "tree_method": "hist"}
        if selected_trees is None:
            fit_data, valid_data = self._purged_time_split(ordered, self.embargo_days_)
            self.validation_start_ = pd.Timestamp(valid_data.decision_date.min())
            self.fit_end_ = pd.Timestamp(fit_data.decision_date.max())
            x_fit, y_fit, weight_fit = self._long_matrix(fit_data, include_target=True)
            x_valid, y_valid, _ = self._long_matrix(valid_data, include_target=True)
            selector = XGBRegressor(n_estimators=STYLE_XGB_MAX_ESTIMATORS,
                                    early_stopping_rounds=STYLE_XGB_EARLY_STOPPING_ROUNDS, **common)
            selector.fit(x_fit, y_fit, sample_weight=weight_fit,
                         eval_set=[(x_valid, y_valid)], verbose=False)
            best_trees = min(STYLE_XGB_MAX_ESTIMATORS, int(selector.best_iteration or 0) + 1)
            validation_predictions = selector.predict(x_valid).reshape(len(valid_data), len(self.styles))
            daily_ic = []
            for position in range(len(valid_data)):
                predicted = validation_predictions[position]
                realized = valid_data.iloc[position][
                    [f"target_{style}_{self.horizon}" for style in self.styles]].to_numpy(dtype=float)
                valid = np.isfinite(predicted) & np.isfinite(realized)
                if (valid.sum() >= 3 and np.std(predicted[valid]) > 1e-12 and
                        np.std(realized[valid]) > 1e-12):
                    daily_ic.append(stats.spearmanr(predicted[valid], realized[valid]).statistic)
            self.validation_rank_ic = float(np.nanmean(daily_ic)) if daily_ic else np.nan
        else:
            best_trees = int(selected_trees)
            self.validation_rank_ic = (float(validation_rank_ic)
                                       if validation_rank_ic is not None else np.nan)
            self.fit_end_ = pd.Timestamp(selection_fit_end) if selection_fit_end is not None else None
            self.validation_start_ = (pd.Timestamp(selection_validation_start)
                                      if selection_validation_start is not None else None)
        self.model = XGBRegressor(n_estimators=best_trees, **common)
        self.model.fit(x_all, y_all, sample_weight=weight_all, verbose=False)
        self.best_iterations = [best_trees]
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("择时XGB尚未拟合")
        x, _, _ = self._long_matrix(data)
        values = self.model.predict(x).reshape(len(data), len(self.styles))
        out = pd.DataFrame({"decision_date": data.decision_date.to_numpy()})
        for index, style in enumerate(self.styles):
            out[style] = values[:, index]
        return out

    def feature_importance_frame(self) -> pd.DataFrame:
        """输出共享XGB对混合特征的归一化重要性。"""
        if self.model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.DataFrame({"feature": self.model_feature_columns,
                             "importance": self.model.feature_importances_}).sort_values(
            "importance", ascending=False, ignore_index=True)
