"""固定PCA风格轮动总入口；交易部分只调用既有BackTestAdapter。"""
from imports import *
from env import (ACTIVE_POOLS, DRAWDOWN_LIMIT, DRAWDOWN_OPTIMIZER_NAMES, MODEL_N_JOBS,
                 OPTIMIZER_NAMES, OUTPUT_ROOT, STYLE_BUY_CONFIRMATIONS,
                 STYLE_EXPLAINED_VARIANCE, STYLE_ICIR_WINDOW, STYLE_MIN_COMPONENTS,
                 STYLE_MODELS, STYLE_RESIDUAL_RIDGE_ALPHA, STYLE_RESIDUAL_WEIGHT,
                 STYLE_VOTE_QUANTILE, STYLE_XGB_MAX_DEPTH, TURNOVER_MAX_RATIO, TURNOVER_OPTIMIZER_NAMES,
                 TURNOVER_SELL_CONFIRMATIONS)
from rolling_ml.backtest_adapter import BacktestAdapter
from rolling_ml.experiment_logger import ExperimentLogger
from rolling_ml.report_generator import ReportGenerator
from optimizers import create_drawdown_optimizer, create_optimizer, create_turnover_optimizer
from style_rotation.config import StyleRuntimeConfig
from style_rotation.pipeline import StyleRotationPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="全市场共享滚动PCA、股票池内独立训练的风格提取与择时")
    parser.add_argument("--test-year", type=int, default=2024)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--test-start", default=None)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--factor-path", default=None)
    parser.add_argument("--features", nargs="+", default=None,
                        help="仅使用指定的因子字段")
    parser.add_argument("--pools", nargs="+", default=ACTIVE_POOLS)
    parser.add_argument("--style-models", nargs="+", choices=STYLE_MODELS,
                        default=["pca_stock_xgboost"])
    parser.add_argument("--optuna-trials", type=int, default=30, help="预留调参次数接口")
    parser.add_argument("--n-jobs", type=int, default=MODEL_N_JOBS)
    parser.add_argument("--window-mode", choices=["expanding", "fixed"], default="expanding")
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--pca-variance", type=float, default=STYLE_EXPLAINED_VARIANCE)
    parser.add_argument("--pca-min-components", type=int, default=STYLE_MIN_COMPONENTS)
    parser.add_argument("--buy-confirmations", type=int, default=STYLE_BUY_CONFIRMATIONS)
    parser.add_argument("--sell-confirmations", type=int, default=TURNOVER_SELL_CONFIRMATIONS,
                        help="惰性换手优化器：达到该看空票数才允许卖出")
    parser.add_argument("--vote-quantile", type=float, default=STYLE_VOTE_QUANTILE)
    parser.add_argument("--residual-weight", type=float, default=STYLE_RESIDUAL_WEIGHT)
    parser.add_argument("--residual-alpha", type=float, default=STYLE_RESIDUAL_RIDGE_ALPHA)
    parser.add_argument("--icir-window", type=int, default=STYLE_ICIR_WINDOW)
    parser.add_argument("--xgb-max-depth", type=int, default=STYLE_XGB_MAX_DEPTH)
    parser.add_argument("--pca-alignment-threshold", type=float, default=.80)
    parser.add_argument("--pca-explained-threshold", type=float, default=.85)
    parser.add_argument("--confidence-rankic-window", type=int, default=60)
    parser.add_argument("--confidence-rankic-threshold", type=float, default=0.0)
    parser.add_argument("--entry-rank", type=int, default=200)
    parser.add_argument("--exit-rank", type=int, default=300)
    parser.add_argument("--holding-count", type=int, default=200)
    parser.add_argument("--rotation-quantile", type=float, default=.10)
    parser.add_argument("--rebalance-frequency", choices=["daily", "alternate", "weekly", "monthly", "quarterly"],
                        default="daily")
    parser.add_argument("--weight-mode", choices=["score", "equal"], default="score")
    parser.add_argument("--pca-fit-mode", choices=["rolling", "fixed"], default="rolling")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--optimizer", choices=OPTIMIZER_NAMES, default="none",
                        help="预测分数进入既有回测前使用的约束优化器")
    parser.add_argument("--turnover-optimizer", choices=TURNOVER_OPTIMIZER_NAMES, default="none")
    parser.add_argument("--drawdown-optimizer", choices=DRAWDOWN_OPTIMIZER_NAMES, default="none")
    parser.add_argument("--max-turnover-ratio", type=float, default=TURNOVER_MAX_RATIO)
    parser.add_argument("--max-drawdown-limit", type=float, default=DRAWDOWN_LIMIT)
    parser.add_argument("--include-costs", action="store_true")
    parser.add_argument("--commission-rate", type=float, default=.0003)
    parser.add_argument("--stamp-duty-rate", type=float, default=.0005)
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
            xgb_max_depth=args.xgb_max_depth,
            pca_alignment_threshold=args.pca_alignment_threshold,
            pca_explained_threshold=args.pca_explained_threshold,
            confidence_rankic_window=args.confidence_rankic_window,
            confidence_rankic_threshold=args.confidence_rankic_threshold,
            entry_rank=args.entry_rank,
            exit_rank=args.exit_rank,
            holding_count=args.holding_count,
            rotation_quantile=args.rotation_quantile,
            rebalance_frequency=args.rebalance_frequency,
            pca_fit_mode=args.pca_fit_mode,
        ).validate()
        direct_pca_xgb = args.style_models == ["pca_stock_xgboost"]
        pipeline_name = ("shared_market_pca_pool_specific_stock_xgboost"
                         if direct_pca_xgb else "shared_market_pca_pool_specific_rotation")
        style_manifest = asdict(style_config)
        if direct_pca_xgb:
            relevant = {"pca_variance", "pca_min_components", "xgb_max_depth",
                        "pca_explained_threshold", "pca_alignment_threshold",
                        "confidence_rankic_window", "confidence_rankic_threshold",
                        "holding_count", "rotation_quantile", "rebalance_frequency"}
            style_manifest = {key: value for key, value in style_manifest.items()
                              if key in relevant}
        manifest_args = vars(args).copy()
        if direct_pca_xgb:
            for key in ("buy_confirmations", "vote_quantile", "residual_weight",
                        "residual_alpha", "icir_window"):
                manifest_args.pop(key, None)
        manifest = {**manifest_args, "style_config": style_manifest,
                    "pipeline": pipeline_name}
        logger.json("config/run_manifest.json", manifest)
        result = StyleRotationPipeline(
            logger, args.test_year, args.style_models, style_config,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            factor_path=args.factor_path, pools=args.pools,
            optuna_trials=args.optuna_trials,
            selected_features=args.features,
            window_mode=args.window_mode, train_years=args.train_years).run()
        rows, curves = [], []
        if not args.skip_backtest:
            adapter = BacktestAdapter()
            for (model, pool), prediction in result["predictions"].groupby(["model", "pool"]):
                logger.status("backtest", .88, f"回测 {model} - {pool}", model=model, pool=pool)
                try:
                    _, metrics, curve = adapter.run(
                        prediction, pool, logger.root / "backtest", model,
                        optimizer_name=args.optimizer,
                        turnover_optimizer_name=args.turnover_optimizer,
                        drawdown_optimizer_name=args.drawdown_optimizer,
                        sell_confirmations=args.sell_confirmations,
                        max_turnover_ratio=args.max_turnover_ratio,
                        max_drawdown_limit=args.max_drawdown_limit,
                        holding_count=style_config.holding_count,
                        selection_mode="top_quantile",
                        rotation_quantile=style_config.rotation_quantile,
                        rebalance_frequency=style_config.rebalance_frequency,
                        weight_mode=args.weight_mode,
                        include_costs=args.include_costs,
                        commission_rate=args.commission_rate,
                        stamp_duty_rate=args.stamp_duty_rate)
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
