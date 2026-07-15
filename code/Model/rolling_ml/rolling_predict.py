from imports import *
from env import (ELASTICNET_ALPHA_GRID, ELASTICNET_L1_RATIO_GRID, ELASTICNET_VALIDATION_FOLDS,
                 RIDGE_ALPHA_GRID, MLConfig, XGB_VALIDATION_FOLDS)
from rolling_ml.data_loader import ExistingDataAdapter
from rolling_ml.dataset_builder import DatasetBuilder
from rolling_ml.experiment_logger import ExperimentLogger
from rolling_ml.label_builder import LabelBuilder
from rolling_ml.metrics import daily_ic, layered_metrics, metric_summary
from rolling_ml.models import create_model
from rolling_ml.optuna_tuner import XGBoostTuner
from rolling_ml.preprocessing import FeaturePreprocessor
from rolling_ml.time_splitter import TimeSplitter


class RollingPredictor:
    def __init__(self, config: MLConfig, logger: ExperimentLogger):
        config.validate()
        self.cfg, self.log = config, logger

    def _linear_params(self, name: str, data: pd.DataFrame, features: list[str], folds) -> dict:
        if name == "linear_regression":
            return {}
        candidates = ([{"alpha": a} for a in RIDGE_ALPHA_GRID] if name == "ridge" else
                      [{"alpha": a, "l1_ratio": r} for a in ELASTICNET_ALPHA_GRID for r in ELASTICNET_L1_RATIO_GRID])
        best, best_score = candidates[0], -np.inf
        tuning_records: list[dict[str, Any]] = []
        for number, params in enumerate(candidates, 1):
            scores = []
            rejected_reason = ""
            self.log.status("annual_tuning", .15 + .18 * (number - 1) / len(candidates),
                            f"{name} 年度参数 {number}/{len(candidates)}", model=name,
                            tuning_current=number, tuning_total=len(candidates), parameters=params)
            for fold_number, (fold_name, train_idx, valid_idx) in enumerate(folds, 1):
                self.log.status("annual_tuning", .15 + .18 *
                                ((number - 1) + (fold_number - 1) / max(len(folds), 1)) / len(candidates),
                                f"{name} 参数 {number}/{len(candidates)}，验证折 {fold_number}/{len(folds)}",
                                model=name, tuning_current=number, tuning_total=len(candidates),
                                fold=fold_number, fold_total=len(folds), fold_name=fold_name, parameters=params)
                train, valid = data.loc[train_idx], data.loc[valid_idx]
                prep = FeaturePreprocessor(True)
                x_train = prep.fit_transform(FeaturePreprocessor.cross_sectional(train, features))
                x_valid = prep.transform(FeaturePreprocessor.cross_sectional(valid, features))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = create_model(name, params).fit(x_train, train.label)
                validation_prediction = model.predict(x_valid)
                if not np.isfinite(validation_prediction).any() or np.nanstd(validation_prediction) <= 1e-12:
                    rejected_reason = "constant_prediction"
                    self.log.info("%s 参数 %d/%d 产生常数预测，跳过剩余验证折：%s",
                                  name, number, len(candidates), params)
                    del x_train, x_valid, model, validation_prediction
                    gc.collect()
                    break
                evaluated = valid[["factor_date", "symbol", "label"]].copy()
                evaluated["prediction"] = validation_prediction
                scores.append(metric_summary(evaluated)["mean_rank_ic"])
                del x_train, x_valid, model, evaluated
                gc.collect()
            finite_scores = [value for value in scores if np.isfinite(value)]
            if not rejected_reason and not finite_scores:
                rejected_reason = "invalid_validation_ic"
            score = float(np.mean(finite_scores)) if finite_scores and not rejected_reason else -np.inf
            tuning_records.append({"model": name, "candidate": number, "total_candidates": len(candidates),
                                   "parameters": json.dumps(params), "mean_rank_ic": score if np.isfinite(score) else None,
                                   "state": "rejected" if rejected_reason else "complete",
                                   "reason": rejected_reason})
            self.log.csv(f"metrics/{name}_tuning_progress.csv", pd.DataFrame(tuning_records))
            self.log.info("%s 参数 %d/%d，验证Mean RankIC=%.6f", name, number, len(candidates), score)
            if score > best_score:
                best, best_score = params, score
        if not np.isfinite(best_score):
            raise RuntimeError(f"{name} 所有候选参数均产生无效或常数预测")
        return best

    def run_year(self, test_year: int = 2024) -> dict[str, Any]:
        started = time.time()
        self.log.info("阶段1/6：加载数据")
        self.log.status("data_loading", .02, "正在加载因子、TWAP和交易日历")
        bundle = ExistingDataAdapter().load(self.cfg.train_start, self.cfg.test_end)
        self.log.info("阶段2/6：构造t0/t1/t2标签和审计")
        self.log.status("label_audit", .08, "正在构造标签并检查未来数据")
        labels = LabelBuilder().build(bundle.real_twap)
        audit = LabelBuilder.audit(labels, bundle.real_twap)
        dataset = DatasetBuilder().build(bundle.factors, labels)
        data, features = dataset.data, dataset.features
        splitter = TimeSplitter(test_year)
        periods = splitter.quarters(bundle.calendar)
        folds = splitter.validation_folds(data, test_year - 1)
        if len(periods) != 4 or len(folds) != 4:
            raise RuntimeError(f"季度切分不完整：预测={len(periods)}，验证={len(folds)}")

        leakage_rows = []
        for split in periods:
            train, test = splitter.training(data, split, self.cfg.train_start, self.cfg.window_mode, self.cfg.train_years), splitter.prediction(data, split)
            splitter.assert_no_leakage(train, test, split)
            leakage_rows.append({"quarter": split.name, "passed": True,
                "max_train_label_exit": train.label_exit_date.max(), "train_cutoff": split.train_cutoff,
                "max_train_factor_date": train.factor_date.max(), "min_prediction_date": test.factor_date.min()})
        self.log.csv("audit/data_quality.csv", bundle.quality)
        self.log.csv("audit/label_alignment_sample.csv", audit)
        self.log.csv("audit/split_summary.csv", pd.DataFrame([asdict(x) for x in periods]))
        self.log.csv("audit/feature_missing_rates.csv", dataset.missing_rates)
        self.log.csv("audit/daily_sample_counts.csv", dataset.daily_counts)
        self.log.json("audit/leakage_check.json", {"passed": all(x["passed"] for x in leakage_rows), "quarters": leakage_rows})
        self.log.json("config/feature_list.json", features)
        self.log.json("config/env_snapshot.json", asdict(self.cfg))

        self.log.info("阶段3/6：年度参数选择（只使用%d年及以前数据）", test_year - 1)
        self.log.status("annual_tuning", .15, "正在使用测试年度之前的数据选择年度参数")
        annual_params: dict[str, dict | None] = {}
        for model_name in self.cfg.models:
            try:
                annual_params[model_name] = (XGBoostTuner(self.log.root / "optuna", self.cfg.optuna_trials,
                                                          self.log.info, self.log.status)
                    .tune(data, features, folds[-XGB_VALIDATION_FOLDS:], test_year) if model_name == "xgboost"
                    else self._linear_params(model_name, data, features,
                        folds[-ELASTICNET_VALIDATION_FOLDS:] if model_name == "elasticnet" else folds))
            except Exception as exc:
                self.log.record_failure(model_name, "annual_tune", exc)
                annual_params[model_name] = None

        self.log.info("阶段4/6：季度滚动训练和样本外预测")
        self.log.status("quarterly_training", .35, "开始季度滚动训练")
        all_predictions, training, live_daily_frames = [], [], []
        total_quarters = max(1, len(self.cfg.models) * len(periods))
        completed_quarters = 0
        for model_name in self.cfg.models:
            params = annual_params.get(model_name)
            if params is None:
                continue
            for split in periods:
                train = test = None
                try:
                    train = splitter.training(data, split, self.cfg.train_start, self.cfg.window_mode, self.cfg.train_years)
                    test = splitter.prediction(data, split)
                    splitter.assert_no_leakage(train, test, split)
                    prep = FeaturePreprocessor(model_name != "xgboost")
                    x_train = prep.fit_transform(FeaturePreprocessor.cross_sectional(train, features))
                    x_test = prep.transform(FeaturePreprocessor.cross_sectional(test, features))
                    model = create_model(model_name, params)
                    train_started = time.time()
                    self.log.status("quarterly_training", .35 + .45 * completed_quarters / total_quarters,
                                    f"正在训练 {model_name} {split.name}", model=model_name, quarter=split.name)
                    model.fit(x_train, train.label)
                    prediction = test[["factor_date", "symbol", "label"]].copy()
                    prediction["prediction"] = model.predict(x_test)
                    prediction["model"], prediction["quarter"] = model_name, split.name
                    model_dir = self.log.root / "models" / model_name / split.name
                    model.save(model_dir / "model.joblib")
                    prep.save(model_dir / "preprocessor.joblib")
                    importance = model.get_feature_importance()
                    if importance is not None and len(importance) == len(features):
                        self.log.csv(f"models/{model_name}/{split.name}/feature_importance.csv",
                                     pd.DataFrame({"feature": features, "importance": importance}))
                    quarter_metric = metric_summary(prediction)
                    # 训练拟合曲线只抽样计算，避免再次对数百万训练样本完整预测。
                    sample_count = min(200_000, len(train))
                    sample_index = np.linspace(0, len(train) - 1, sample_count, dtype=int)
                    train_evaluation = train.iloc[sample_index][["factor_date", "symbol", "label"]].copy()
                    train_evaluation["prediction"] = model.predict(x_train[sample_index])
                    train_metric = metric_summary(train_evaluation)
                    training.append({"model": model_name, "quarter": split.name,
                        "train_start": train.factor_date.min(), "train_cutoff": split.train_cutoff,
                        "prediction_start": split.prediction_start, "prediction_end": split.prediction_end,
                        "maximum_factor_date_in_train": train.factor_date.max(),
                        "maximum_label_exit_date_in_train": train.label_exit_date.max(),
                        "minimum_prediction_date": test.factor_date.min(), "training_sample_count": len(train),
                        "prediction_sample_count": len(test), "feature_count": len(features),
                        "parameters": json.dumps(params), "model_path": str(model_dir / "model.joblib"),
                        "preprocessor_path": str(model_dir / "preprocessor.joblib"),
                        "training_duration": time.time() - train_started,
                        "train_rank_ic": train_metric["mean_rank_ic"],
                        "quarter_rank_ic": quarter_metric["mean_rank_ic"], "quarter_icir": quarter_metric["icir"]})
                    self.log.csv("metrics/training_records.csv", pd.DataFrame(training))
                    prediction.to_parquet(self.log.root / "predictions/quarterly" / f"{model_name}_{split.name}.parquet", index=False)
                    all_predictions.append(prediction)
                    live_daily = daily_ic(prediction.dropna(subset=["label"]))
                    live_daily["model"] = model_name
                    live_daily_frames.append(live_daily)
                    self.log.csv("metrics/daily_rankic.csv", pd.concat(live_daily_frames, ignore_index=True))
                    self.log.info("完成 %s %s：训练%d，预测%d", model_name, split.name, len(train), len(test))
                    completed_quarters += 1
                    self.log.status("quarterly_training", .35 + .45 * completed_quarters / total_quarters,
                                    f"已完成 {model_name} {split.name}", model=model_name, quarter=split.name)
                    del x_train, x_test, model, prep
                    gc.collect()
                except Exception as exc:
                    train_range = "" if train is None else f"{train.factor_date.min()} ~ {train.factor_date.max()}"
                    self.log.record_failure(model_name, split.name, exc, params, train_range)

        predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
        records = pd.DataFrame(training)
        self.log.csv("metrics/training_records.csv", records)
        if predictions.empty:
            raise RuntimeError("没有模型成功生成预测，请查看 logs/failed_tasks.csv")
        predictions.to_parquet(self.log.root / "predictions/all_predictions.parquet", index=False)

        self.log.info("阶段5/6：计算日度、季度和年度指标")
        self.log.status("metrics", .82, "正在计算每日、季度和年度IC")
        daily_frames = []
        for model_name, group in predictions.groupby("model"):
            daily = daily_ic(group.dropna(subset=["label"]))
            daily["model"] = model_name
            daily_frames.append(daily)
            factor_file = group[["factor_date", "symbol", "prediction"]].rename(
                columns={"factor_date": "date", "symbol": "ticker", "prediction": model_name}
            ).set_index(["ticker", "date"]).sort_index()
            factor_file.to_parquet(self.log.root / "predictions/model_prediction_files" / f"{model_name}_predictions.parquet")
        daily_frame = pd.concat(daily_frames, ignore_index=True)
        layered = layered_metrics(predictions)
        summary = layered[layered.level == "full_test"].reset_index(drop=True)
        self.log.csv("metrics/daily_rankic.csv", daily_frame)
        self.log.csv("metrics/quarterly_rankic.csv", layered[layered.level == "quarter"])
        self.log.csv("metrics/annual_rankic.csv", layered[layered.level == "year"])
        self.log.csv("metrics/model_summary.csv", summary)
        self.log.json("config/run_manifest.json", {"run_id": self.log.run_id,
            "duration": time.time() - started, "models": list(self.cfg.models), "pools": list(self.cfg.pools),
            "annual_parameters": annual_params, "data_audit_passed": True})
        self.log.status("model_complete", .88, "模型训练和预测完成，准备回测")
        return {"predictions": predictions, "daily_metrics": daily_frame, "summary": summary,
                "layered_metrics": layered, "training_records": records, "features": features,
                "params": annual_params, "duration": time.time() - started}
