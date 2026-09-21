from imports import *
from env import (MODEL_N_JOBS, STYLE_XGB_EARLY_STOPPING_ROUNDS,
                 STYLE_XGB_MAX_ESTIMATORS, STYLE_XGB_VALIDATION_EMBARGO_DAYS,
                 XGB_OBJECTIVE, OPTUNA_N_JOBS, OPTUNA_N_STARTUP_TRIALS, OPTUNA_SEED)
from rolling_ml.metrics import daily_ic


def _embargo_observations(dates: pd.DatetimeIndex, trading_days: int) -> int:
    """Convert a trading-day embargo into observations of the factor calendar."""
    if trading_days <= 0:
        return 0
    ordered = pd.DatetimeIndex(dates).drop_duplicates().sort_values()
    if len(ordered) < 2:
        return max(1, int(trading_days))
    gaps = pd.Series(ordered[1:] - ordered[:-1]).dt.total_seconds() / 86400
    median_calendar_days = float(gaps[gaps > 0].median())
    if not np.isfinite(median_calendar_days):
        return max(1, int(trading_days))
    trading_days_per_observation = max(1.0, median_calendar_days * 252 / 365.25)
    return max(1, int(np.ceil(trading_days / trading_days_per_observation)))


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


class PCAStockXGBoost:
    """Predict stock returns directly from aligned PCA exposures."""

    def __init__(self, styles: list[str], params: dict | None = None):
        if XGBRegressor is None:
            raise ImportError("xgboost未安装")
        self.styles, self.params = styles, params or {}
        self.model: Any = None
        self.validation_rank_ic = np.nan
        self.best_trees = 0
        self.fit_end_: pd.Timestamp | None = None
        self.validation_start_: pd.Timestamp | None = None
        self.embargo_days_ = 0
        self.embargo_observations_ = 0

    def _matrix(self, data: pd.DataFrame) -> np.ndarray:
        return data[self.styles].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(
            dtype=np.float32, copy=False)

    def _common(self) -> dict[str, Any]:
        return {"max_depth": self.params.get("max_depth", 2),
                "learning_rate": self.params.get("learning_rate", .03),
                "min_child_weight": self.params.get("min_child_weight", 1.0),
                "gamma": self.params.get("gamma", 0.0),
                "reg_alpha": self.params.get("reg_alpha", 0.0),
                "reg_lambda": self.params.get("reg_lambda", 1.0),
                "subsample": self.params.get("subsample", .75),
                "colsample_bytree": self.params.get("colsample_bytree", .70),
                "max_bin": self.params.get("max_bin", 256),
                "n_jobs": MODEL_N_JOBS, "random_state": 42,
                "objective": XGB_OBJECTIVE, "tree_method": "hist"}

    def fit(self, data: pd.DataFrame, selected_trees: int | None = None,
            validation_rank_ic: float | None = None,
            selection_fit_end: pd.Timestamp | str | None = None,
            selection_validation_start: pd.Timestamp | str | None = None):
        valid = np.isfinite(pd.to_numeric(data["label"], errors="coerce"))
        if "sample_weight" in data:
            valid &= np.isfinite(pd.to_numeric(data["sample_weight"], errors="coerce"))
        ordered = data.loc[valid].sort_values(
            ["factor_date", "symbol"]).reset_index(drop=True)
        if ordered.empty:
            raise ValueError("PCA股票XGB没有有效训练标签")
        common = self._common()
        if selected_trees is None:
            dates = pd.DatetimeIndex(ordered.factor_date.drop_duplicates().sort_values())
            self.embargo_observations_ = _embargo_observations(dates, self.embargo_days_)
            valid_position = max(self.embargo_observations_ + 1,
                                 min(len(dates) - 1, int(len(dates) * .80)))
            fit_end_position = valid_position - self.embargo_observations_
            if fit_end_position < 1 or valid_position >= len(dates):
                raise ValueError(
                    "PCA股票XGB样本不足，无法建立隔离验证集："
                    f"有效因子日期={len(dates)}，{self.embargo_days_}个交易日"
                    f"折算为{self.embargo_observations_}个因子观测期")
            fit_dates, valid_dates = dates[:fit_end_position], dates[valid_position:]
            fit = ordered[ordered.factor_date.isin(fit_dates)]
            valid = ordered[ordered.factor_date.isin(valid_dates)]
            if fit.empty or valid.empty:
                raise ValueError(
                    f"PCA股票XGB隔离后为空：拟合={len(fit)}，验证={len(valid)}")
            selector = XGBRegressor(
                n_estimators=STYLE_XGB_MAX_ESTIMATORS,
                early_stopping_rounds=STYLE_XGB_EARLY_STOPPING_ROUNDS, **common)
            selector.fit(self._matrix(fit), fit.label,
                         sample_weight=fit.sample_weight,
                         eval_set=[(self._matrix(valid), valid.label)], verbose=False)
            selected_trees = min(
                STYLE_XGB_MAX_ESTIMATORS, int(selector.best_iteration or 0) + 1)
            evaluated = valid[["factor_date", "symbol", "label"]].copy()
            evaluated["prediction"] = selector.predict(self._matrix(valid))
            self.validation_rank_ic = float(daily_ic(evaluated).rank_ic.mean())
            self.fit_end_ = pd.Timestamp(fit.factor_date.max())
            self.validation_start_ = pd.Timestamp(valid.factor_date.min())
        else:
            self.validation_rank_ic = float(validation_rank_ic or 0.0)
            self.fit_end_ = (pd.Timestamp(selection_fit_end)
                             if selection_fit_end is not None else None)
            self.validation_start_ = (pd.Timestamp(selection_validation_start)
                                      if selection_validation_start is not None else None)
        self.best_trees = int(selected_trees)
        self.model = XGBRegressor(n_estimators=self.best_trees, **common)
        self.model.fit(self._matrix(ordered), ordered.label,
                       sample_weight=ordered.sample_weight, verbose=False)
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("PCA股票XGB尚未拟合")
        return np.asarray(self.model.predict(self._matrix(data)), dtype=float)

    def feature_importance_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"feature": self.styles,
                             "importance": self.model.feature_importances_}).sort_values(
            "importance", ascending=False, ignore_index=True)


class PCAStockXGBoostTuner:
    """Use the ordinary XGBoost Optuna search on PCA stock exposures."""

    def __init__(self, run_root: Path, n_trials: int = 30, progress=None, status=None):
        self.root, self.n_trials = Path(run_root), int(n_trials)
        self.progress, self.status = progress, status

    @staticmethod
    def _split(data: pd.DataFrame, embargo_days: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
        valid = np.isfinite(pd.to_numeric(data["label"], errors="coerce"))
        if "sample_weight" in data:
            valid &= np.isfinite(pd.to_numeric(data["sample_weight"], errors="coerce"))
        ordered = data.loc[valid].sort_values(
            ["factor_date", "symbol"]).reset_index(drop=True)
        dates = pd.DatetimeIndex(ordered.factor_date.drop_duplicates().sort_values())
        embargo_observations = _embargo_observations(dates, embargo_days)
        valid_position = max(embargo_observations + 1,
                             min(len(dates) - 1, int(len(dates) * .80)))
        fit_end_position = valid_position - embargo_observations
        if fit_end_position < 1 or valid_position >= len(dates):
            raise ValueError(
                "PCA股票XGB历史不足以构造无泄漏时间验证集："
                f"有效因子日期={len(dates)}")
        fit_dates, valid_dates = dates[:fit_end_position], dates[valid_position:]
        fit = ordered[ordered.factor_date.isin(fit_dates)]
        valid = ordered[ordered.factor_date.isin(valid_dates)]
        if "label_exit_date" in fit.columns and len(valid_dates):
            fit = fit[pd.to_datetime(fit.label_exit_date) < valid_dates.min()]
        if fit.empty or valid.empty:
            raise ValueError(
                "PCA股票XGB历史不足以构造无泄漏时间验证集："
                f"有效因子日期={len(dates)}，拟合={len(fit)}，验证={len(valid)}")
        return fit, valid

    def tune(self, data: pd.DataFrame, styles: list[str], year: int, pool: str,
             embargo_days: int = 0) -> dict[str, Any]:
        if optuna is None or XGBRegressor is None:
            raise ImportError("PCA XGBoost调参需要安装 optuna 和 xgboost")
        self.root.mkdir(parents=True, exist_ok=True)
        base = PCAStockXGBoost(styles)
        # PCA stock XGB has one label per factor observation.  Its labels are
        # already filtered by label_exit_date in the pipeline, so applying the
        # legacy 20-trading-day embargo here would incorrectly remove one or
        # more monthly observations.  Keep this split purely chronological.
        fit, valid = self._split(data, embargo_days)
        fold_records: list[dict[str, Any]] = []

        def objective(trial):
            if self.status:
                self.status("pca_stock_xgb", .20,
                            f"PCA XGBoost {pool} Trial {trial.number + 1}/{self.n_trials}",
                            model="style_rotation", pool=pool,
                            trial=trial.number + 1, trial_total=self.n_trials)
            params = {
                "learning_rate": trial.suggest_float("learning_rate", .01, .05, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 5),
                "min_child_weight": trial.suggest_float("min_child_weight", 10, 150, log=True),
                "gamma": trial.suggest_float("gamma", 1e-4, 1, log=True),
                "subsample": trial.suggest_float("subsample", .6, .95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", .5, .95),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 20, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1, 100, log=True),
                "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
            }
            common = PCAStockXGBoost(styles, params)._common()
            selector = XGBRegressor(n_estimators=STYLE_XGB_MAX_ESTIMATORS, **common)
            selector.fit(base._matrix(fit), fit.label, sample_weight=fit.sample_weight,
                         verbose=False)
            evaluated = valid[["factor_date", "symbol", "label"]].copy()
            evaluated["prediction"] = selector.predict(base._matrix(valid))
            from rolling_ml.metrics import metric_summary, model_selection_score
            validation_metrics = metric_summary(evaluated)
            score = float(model_selection_score(validation_metrics))
            if not np.isfinite(score):
                trial.set_user_attr("invalid_reason", "non_finite_validation_rank_ic")
                raise optuna.TrialPruned("PCA XGBoost验证RankIC无效")
            trial.set_user_attr("best_iteration", STYLE_XGB_MAX_ESTIMATORS - 1)
            trial.set_user_attr("rank_ic", float(validation_metrics["mean_rank_ic"]))
            trial.set_user_attr("pearson_ic", float(validation_metrics["pearson_ic"]))
            trial.set_user_attr("top_bottom_return", float(validation_metrics["top_bottom_return"]))
            trial.set_user_attr("selection_score", score)
            fold_records.append({"trial_number": trial.number, "pool": pool,
                                 "rank_ic": float(validation_metrics["mean_rank_ic"]),
                                 "pearson_ic": float(validation_metrics["pearson_ic"]),
                                 "top_bottom_return": float(validation_metrics["top_bottom_return"]),
                                 "selection_score": score, "train_samples": len(fit),
                                 "valid_samples": len(valid),
                                 "best_iteration": STYLE_XGB_MAX_ESTIMATORS - 1})
            return score

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study_name = f"{year}_pca_stock_xgboost_{pool}"
        study = optuna.create_study(
            direction="maximize", sampler=TPESampler(
                n_startup_trials=OPTUNA_N_STARTUP_TRIALS, multivariate=True, seed=OPTUNA_SEED),
            study_name=study_name,
            storage=f"sqlite:///{(self.root / f'{study_name}.db').as_posix()}",
            load_if_exists=True)
        completed = sum(item.state.name == "COMPLETE" for item in study.trials)
        remaining = max(0, self.n_trials - completed)
        if remaining:
            def persist_progress(current_study, _trial):
                current_study.trials_dataframe().to_csv(
                    self.root / f"{study_name}_trials_live.csv", index=False, encoding="utf-8-sig")
            study.optimize(objective, n_trials=remaining, n_jobs=OPTUNA_N_JOBS,
                           gc_after_trial=True, callbacks=[persist_progress])
        try:
            best = study.best_trial
        except ValueError as exc:
            raise RuntimeError(f"PCA XGBoost没有完成有效Optuna Trial：{pool}") from exc
        trials = study.trials_dataframe()
        trials.to_csv(self.root / f"{study_name}_trials.csv", index=False, encoding="utf-8-sig")
        best_trees = STYLE_XGB_MAX_ESTIMATORS
        params = {**study.best_params, "n_estimators": best_trees}
        (self.root / f"{study_name}_best_params.json").write_text(
            json.dumps(params, indent=2), encoding="utf-8")
        pd.DataFrame(fold_records).to_csv(
            self.root / f"{study_name}_fold_results.csv", index=False, encoding="utf-8-sig")
        return {"params": params, "trees": best_trees,
                "validation_rank_ic": float(best.value),
                "fit_end": str(fit.factor_date.max()),
                "validation_start": str(valid.factor_date.min())}


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
        embargo_observations = _embargo_observations(dates, embargo_days)
        if len(dates) <= embargo_observations + 2:
            raise ValueError(f"择时XGB样本不足，无法留出{embargo_days}个交易日隔离期")
        valid_position = max(embargo_observations + 1,
                             min(len(dates) - 1, int(len(dates) * .80)))
        fit_end_position = valid_position - embargo_observations
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
