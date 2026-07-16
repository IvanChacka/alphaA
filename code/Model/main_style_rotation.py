"""固定PCA风格轮动总入口；交易部分只调用既有BackTestAdapter。"""
from imports import *
from env import (ACTIVE_POOLS, DRAWDOWN_LIMIT, DRAWDOWN_OPTIMIZER_NAMES, MODEL_N_JOBS,
                 OPTIMIZER_NAMES, OUTPUT_ROOT, STYLE_BUY_CONFIRMATIONS,
                 STYLE_EXPLAINED_VARIANCE, STYLE_ICIR_WINDOW, STYLE_MIN_COMPONENTS,
                 STYLE_MODELS, STYLE_RESIDUAL_RIDGE_ALPHA, STYLE_RESIDUAL_WEIGHT,
                 STYLE_VOTE_QUANTILE, TURNOVER_MAX_RATIO, TURNOVER_OPTIMIZER_NAMES,
                 TURNOVER_SELL_CONFIRMATIONS)
from rolling_ml.backtest_adapter import BacktestAdapter
from rolling_ml.experiment_logger import ExperimentLogger
from rolling_ml.report_generator import ReportGenerator
from optimizers import create_drawdown_optimizer, create_optimizer, create_turnover_optimizer
from style_rotation.config import StyleRuntimeConfig
from style_rotation.pipeline import StyleRotationPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="历史区间一次拟合、测试期永久冻结的PCA风格提取与择时")
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--pools", nargs="+", default=ACTIVE_POOLS)
    parser.add_argument("--style-models", nargs="+", choices=STYLE_MODELS, default=STYLE_MODELS)
    parser.add_argument("--optuna-trials", type=int, default=30, help="预留调参次数接口")
    parser.add_argument("--n-jobs", type=int, default=MODEL_N_JOBS)
    parser.add_argument("--pca-variance", type=float, default=STYLE_EXPLAINED_VARIANCE)
    parser.add_argument("--pca-min-components", type=int, default=STYLE_MIN_COMPONENTS)
    parser.add_argument("--buy-confirmations", type=int, default=STYLE_BUY_CONFIRMATIONS)
    parser.add_argument("--sell-confirmations", type=int, default=TURNOVER_SELL_CONFIRMATIONS,
                        help="惰性换手优化器：达到该看空票数才允许卖出")
    parser.add_argument("--vote-quantile", type=float, default=STYLE_VOTE_QUANTILE)
    parser.add_argument("--residual-weight", type=float, default=STYLE_RESIDUAL_WEIGHT)
    parser.add_argument("--residual-alpha", type=float, default=STYLE_RESIDUAL_RIDGE_ALPHA)
    parser.add_argument("--icir-window", type=int, default=STYLE_ICIR_WINDOW)
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--optimizer", choices=OPTIMIZER_NAMES, default="none",
                        help="预测分数进入既有回测前使用的约束优化器")
    parser.add_argument("--turnover-optimizer", choices=TURNOVER_OPTIMIZER_NAMES, default="none")
    parser.add_argument("--drawdown-optimizer", choices=DRAWDOWN_OPTIMIZER_NAMES, default="none")
    parser.add_argument("--max-turnover-ratio", type=float, default=TURNOVER_MAX_RATIO)
    parser.add_argument("--max-drawdown-limit", type=float, default=DRAWDOWN_LIMIT)
    parser.add_argument("--resume-run")
    return parser.parse_args()


def main():
    args = parse_args()
    run_id = args.resume_run or datetime.now().strftime("style_run_%Y%m%d_%H%M%S")
    logger = ExperimentLogger(OUTPUT_ROOT, run_id)
    try:
        if not args.skip_backtest:
            create_optimizer(args.optimizer)
            create_turnover_optimizer(args.turnover_optimizer, args.sell_confirmations,
                                      args.max_turnover_ratio)
            create_drawdown_optimizer(args.drawdown_optimizer, args.max_drawdown_limit)
        style_config = StyleRuntimeConfig(
            pca_variance=args.pca_variance,
            pca_min_components=args.pca_min_components,
            buy_confirmations=args.buy_confirmations,
            vote_quantile=args.vote_quantile,
            residual_weight=args.residual_weight,
            residual_alpha=args.residual_alpha,
            icir_window=args.icir_window,
        ).validate()
        manifest = {**vars(args), "style_config": asdict(style_config),
                    "pipeline": "fixed_pca_style_rotation"}
        logger.json("config/run_manifest.json", manifest)
        result = StyleRotationPipeline(logger, args.test_year, args.style_models, style_config).run()
        rows, curves = [], []
        if not args.skip_backtest:
            adapter = BacktestAdapter()
            for model, prediction in result["predictions"].groupby("model"):
                for pool in args.pools:
                    logger.status("backtest", .88, f"回测 {model} - {pool}", model=model, pool=pool)
                    try:
                        _, metrics, curve = adapter.run(
                            prediction, pool, logger.root / "backtest", model,
                            optimizer_name=args.optimizer,
                            turnover_optimizer_name=args.turnover_optimizer,
                            drawdown_optimizer_name=args.drawdown_optimizer,
                            sell_confirmations=args.sell_confirmations,
                            max_turnover_ratio=args.max_turnover_ratio,
                            max_drawdown_limit=args.max_drawdown_limit)
                        rows.append({"model": model, "pool": pool, **metrics})
                        curve["model"], curve["pool"] = model, pool
                        curves.append(curve)
                    except Exception as exc:
                        logger.record_failure(model, f"backtest_{pool}", exc)
        backtests = pd.DataFrame(rows)
        curve_frame = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
        logger.csv("backtest/results/summary.csv", backtests)
        report = ReportGenerator().generate(logger.root / "report/index.html", logger.run_id, manifest,
            result["daily_metrics"], result["summary"], result["training_records"], logger.failures,
            pd.DataFrame(), backtests, curve_frame, {}, pd.DataFrame(),
            pd.read_csv(logger.root / "timing/xgb_feature_importance.csv")
            if (logger.root / "timing/xgb_feature_importance.csv").exists() else pd.DataFrame(), manifest,
            pca_style_metrics=result["pca_style_metrics"])
        logger.status("complete", 1, "风格模型与回测完成", report=str(report))
        print(f"运行目录：{logger.root}\n报告：{report}")
    except BaseException as exc:
        logger.record_failure("style_rotation", "fatal", exc)
        raise
    finally:
        logger.close()


if __name__ == "__main__":
    main()
