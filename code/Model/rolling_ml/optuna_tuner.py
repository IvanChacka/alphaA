from imports import *
from env import OPTUNA_N_JOBS, OPTUNA_N_STARTUP_TRIALS, OPTUNA_N_TRIALS, OPTUNA_SEED, XGB_N_ESTIMATORS
from rolling_ml.metrics import metric_summary
from rolling_ml.models.xgb_model import XGBoostModel
from rolling_ml.preprocessing import FeaturePreprocessor


class XGBoostTuner:
    """年度TPE调参；SQLite持久化，全年只调用一次。"""

    def __init__(self, run_root: Path, n_trials: int = OPTUNA_N_TRIALS, progress=None, status=None):
        self.root, self.n_trials, self.progress, self.status = Path(run_root), n_trials, progress, status

    def tune(self, data: pd.DataFrame, features: list[str], folds, year: int = 2024) -> dict:
        if optuna is None or XGBRegressor is None:
            raise ImportError("XGBoost调参需要安装 optuna 和 xgboost")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        self.root.mkdir(parents=True, exist_ok=True)
        fold_records: list[dict[str, Any]] = []

        def objective(trial):
            started = time.time()
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
            evaluated_folds, validation_scores, training_scores, best_iterations = [], [], [], []
            try:
                for fold_number, (name, train_idx, valid_idx) in enumerate(folds, 1):
                    if self.status:
                        self.status("annual_tuning", .15,
                                    f"XGBoost Trial {trial.number + 1}/{self.n_trials}，验证折 {fold_number}/{len(folds)}",
                                    model="xgboost", trial=trial.number + 1, trial_total=self.n_trials,
                                    fold=fold_number, fold_total=len(folds))
                    train, valid = data.loc[train_idx], data.loc[valid_idx]
                    prep = FeaturePreprocessor(scale=False)
                    x_train = prep.fit_transform(FeaturePreprocessor.cross_sectional(train, features))
                    x_valid = prep.transform(FeaturePreprocessor.cross_sectional(valid, features))
                    model = XGBoostModel(params).fit(x_train, train.label, x_valid, valid.label)
                    valid_eval = valid[["factor_date", "symbol", "label"]].copy()
                    valid_eval["prediction"] = model.predict(x_valid)
                    train_eval = train[["factor_date", "symbol", "label"]].copy()
                    train_eval["prediction"] = model.predict(x_train)
                    valid_metric, train_metric = metric_summary(valid_eval), metric_summary(train_eval)
                    validation_scores.append(valid_metric["mean_rank_ic"])
                    training_scores.append(train_metric["mean_rank_ic"])
                    best_iterations.append(model.best_iteration)
                    evaluated_folds.append(valid_eval)
                    fold_records.append({"trial_number": trial.number, "fold": name,
                        "rank_ic": valid_metric["mean_rank_ic"], "icir": valid_metric["icir"],
                        "train_rank_ic": train_metric["mean_rank_ic"], "train_samples": len(train),
                        "valid_samples": len(valid), "feature_count": len(features),
                        "best_iteration": model.best_iteration})
                    del x_train, x_valid, model, train_eval, valid_eval
                    gc.collect()
                combined = metric_summary(pd.concat(evaluated_folds, ignore_index=True))
                trial.set_user_attr("rank_ic_std", float(combined["rank_ic_std"]))
                trial.set_user_attr("icir", float(combined["icir"]))
                trial.set_user_attr("train_rank_ic", float(np.nanmean(training_scores)))
                trial.set_user_attr("best_iteration", int(np.nanmedian(best_iterations)))
                trial.set_user_attr("duration_seconds", time.time() - started)
                for index, score in enumerate(validation_scores, 1):
                    trial.set_user_attr(f"fold_{index}_rank_ic", float(score))
                return float(combined["mean_rank_ic"])
            except Exception as exc:
                trial.set_user_attr("exception", f"{type(exc).__name__}: {exc}")
                raise

        sampler = TPESampler(n_startup_trials=OPTUNA_N_STARTUP_TRIALS, multivariate=True, seed=OPTUNA_SEED)
        study = optuna.create_study(direction="maximize", sampler=sampler, study_name=f"{year}_xgboost",
            storage=f"sqlite:///{(self.root / f'{year}_xgboost_study.db').as_posix()}", load_if_exists=True)
        completed = sum(trial.state.name == "COMPLETE" for trial in study.trials)
        remaining = max(0, self.n_trials - completed)
        if self.progress: self.progress("Optuna开始：%s，已完成%d，剩余%d", study.study_name, completed, remaining)
        if remaining:
            def persist_progress(current_study, _trial):
                current_study.trials_dataframe().to_csv(
                    self.root / f"{year}_trials_live.csv", index=False, encoding="utf-8-sig")
                if self.status:
                    complete = sum(item.state.name == "COMPLETE" for item in current_study.trials)
                    self.status("annual_tuning", .15 + .18 * complete / max(self.n_trials, 1),
                                f"XGBoost已完成 {complete}/{self.n_trials} 个Trial",
                                model="xgboost", trial=complete, trial_total=self.n_trials)
            study.optimize(objective, n_trials=remaining, n_jobs=OPTUNA_N_JOBS,
                           gc_after_trial=True, callbacks=[persist_progress])
        trials = study.trials_dataframe()
        trials.to_csv(self.root / f"{year}_trials.csv", index=False, encoding="utf-8-sig")
        trials.sort_values("value", ascending=False).head(20).to_csv(
            self.root / f"{year}_top20_trials.csv", index=False, encoding="utf-8-sig")
        fold_frame = pd.DataFrame(fold_records)
        best_folds = fold_frame[fold_frame.trial_number == study.best_trial.number]
        stored_iteration = study.best_trial.user_attrs.get("best_iteration")
        best_iteration = (int(np.nanmedian(best_folds.best_iteration)) + 1 if not best_folds.empty
                          else int(stored_iteration) + 1 if stored_iteration is not None else XGB_N_ESTIMATORS)
        best_iteration = min(best_iteration, XGB_N_ESTIMATORS)
        best_params = {**study.best_params, "n_estimators": best_iteration}
        for name, value in [("best_params", best_params), ("best_iterations", {"n_estimators": best_iteration})]:
            (self.root / f"{year}_{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
        best_folds.to_csv(self.root / f"{year}_best_fold_results.csv", index=False, encoding="utf-8-sig")
        try:
            importance = optuna.importance.get_param_importances(study)
            pd.DataFrame(importance.items(), columns=["parameter", "importance"]).to_csv(
                self.root / f"{year}_parameter_importance.csv", index=False, encoding="utf-8-sig")
        except Exception:
            pd.DataFrame(columns=["parameter", "importance"]).to_csv(
                self.root / f"{year}_parameter_importance.csv", index=False, encoding="utf-8-sig")
        return best_params
