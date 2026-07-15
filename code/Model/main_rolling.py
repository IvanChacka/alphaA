"""季度滚动机器学习总入口。"""
from imports import *
from env import (ACTIVE_POOLS, LINEAR_MODELS, MODEL_N_JOBS, NONLINEAR_MODELS,
                 MLConfig, OPTIMIZER_NAMES, OPTUNA_N_TRIALS)
from optimizers import create_optimizer
from rolling_ml.backtest_adapter import BacktestAdapter
from rolling_ml.experiment_logger import ExperimentLogger
from rolling_ml.report_generator import ReportGenerator
from rolling_ml.rolling_predict import RollingPredictor


def parse_args():
    parser = argparse.ArgumentParser(description="季度滚动机器学习预测")
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--pools", nargs="+", default=ACTIVE_POOLS)
    parser.add_argument("--models", nargs="+", default=LINEAR_MODELS + NONLINEAR_MODELS)
    parser.add_argument("--optuna-trials", type=int, default=OPTUNA_N_TRIALS)
    parser.add_argument("--n-jobs", type=int, default=MODEL_N_JOBS)
    parser.add_argument("--window-mode", choices=["expanding", "fixed"], default="expanding")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--optimizer", choices=OPTIMIZER_NAMES, default="none",
                        help="预测分数进入既有回测前使用的约束优化器")
    parser.add_argument("--resume-run", help="复用指定run目录并续跑Optuna，例如 run_20240701_120000")
    return parser.parse_args()


def _read_optional(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def main() -> None:
    args = parse_args()
    cfg = MLConfig(models=tuple(args.models), pools=tuple(args.pools), optuna_trials=args.optuna_trials,
                   n_jobs=args.n_jobs, window_mode=args.window_mode,
                   test_start=f"{args.test_year}-01-01", test_end=f"{args.test_year}-12-31")
    logger = ExperimentLogger(cfg.output_root, args.resume_run)
    try:
        if not args.skip_backtest:
            create_optimizer(args.optimizer)
        result = RollingPredictor(cfg, logger).run_year(args.test_year)
        backtest_rows, curve_frames = [], []
        detail_frames = {"account": [], "orders": [], "trades": [], "holdings": [], "adjustments": []}
        if not args.skip_backtest:
            logger.info("阶段6/6：调用现有回测框架")
            logger.status("backtest", .90, "开始调用原回测框架")
            adapter = BacktestAdapter()
            for model_name, prediction in result["predictions"].groupby("model"):
                for pool in cfg.pools:
                    try:
                        logger.status("backtest", .92, f"正在回测 {model_name} - {pool}", model=model_name, pool=pool)
                        bt_result, metrics, curve = adapter.run(
                            prediction, pool, logger.root / "backtest", model_name,
                            optimizer_name=args.optimizer)
                        backtest_rows.append({"model": model_name, "pool": pool, **metrics})
                        curve["model"], curve["pool"] = model_name, pool
                        curve_frames.append(curve)
                        for key, frame in [("account", bt_result.account.reset_index()), ("orders", bt_result.orders),
                                           ("trades", bt_result.trades), ("holdings", bt_result.holdings)]:
                            tagged = frame.copy(); tagged["model"], tagged["pool"] = model_name, pool
                            detail_frames[key].append(tagged)
                        adjustment_path = logger.root / "backtest/adjustments" / f"{model_name}_{pool}_adjustments.csv"
                        adjustment = _read_optional(adjustment_path)
                        if not adjustment.empty:
                            adjustment["model"], adjustment["pool"] = model_name, pool
                            detail_frames["adjustments"].append(adjustment)
                    except Exception as exc:
                        logger.record_failure(model_name, f"backtest_{pool}", exc)
        backtests = pd.DataFrame(backtest_rows)
        curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
        logger.csv("backtest/results/summary.csv", backtests)
        details = {key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                   for key, frames in detail_frames.items()}
        year = args.test_year
        trials = _read_optional(logger.root / "optuna" / f"{year}_trials.csv")
        parameter_importance = _read_optional(logger.root / "optuna" / f"{year}_parameter_importance.csv")
        importance_files = list((logger.root / "models").glob("*/*/feature_importance.csv"))
        importance = pd.concat([pd.read_csv(file).assign(path=str(file.relative_to(logger.root)))
                                for file in importance_files], ignore_index=True) if importance_files else pd.DataFrame()
        manifest = json.loads((logger.root / "config/run_manifest.json").read_text(encoding="utf-8"))
        manifest["optimizer"] = args.optimizer
        logger.json("config/run_manifest.json", manifest)
        report = ReportGenerator().generate(logger.root / "report/index.html", logger.run_id,
            {**asdict(cfg), "optimizer": args.optimizer},
            result["daily_metrics"], result["summary"], result["training_records"], logger.failures,
            trials, backtests, curves, details, parameter_importance, importance, manifest)
        print(f"运行目录：{logger.root}")
        print(result["summary"].to_string(index=False))
        if result["params"].get("xgboost"):
            print("XGBoost最优参数：", result["params"]["xgboost"])
        if not backtests.empty:
            columns = [c for c in ["model", "pool", "absolute_annual_return", "absolute_sharpe"] if c in backtests]
            print("回测结果：\n", backtests[columns].to_string(index=False))
        print(f"HTML报告：{report}")
        logger.status("complete", 1.0, "训练、回测和报告全部完成", report=str(report))
    except BaseException as exc:
        logger.record_failure("pipeline", "fatal", exc)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
