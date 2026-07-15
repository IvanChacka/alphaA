from imports import *
from env import BACKTEST_DIR
from optimizers import create_optimizer


class BacktestAdapter:
    """把t0预测注入现有BacktestEngine，不修改、不复制回测业务逻辑。"""

    def __init__(self):
        if str(BACKTEST_DIR) not in sys.path:
            sys.path.insert(0, str(BACKTEST_DIR))

    @staticmethod
    def prediction_wide(predictions: pd.DataFrame) -> pd.DataFrame:
        required = {"factor_date", "symbol", "prediction"}
        if not required.issubset(predictions.columns):
            raise ValueError(f"预测缺少字段：{required - set(predictions.columns)}")
        frame = predictions.copy()
        frame.symbol = frame.symbol.astype(str).str.zfill(6)
        return frame.pivot_table(index="factor_date", columns="symbol", values="prediction", aggfunc="last").sort_index()

    @staticmethod
    def pool_ic_statistics(values: list[float]) -> dict[str, float]:
        """从逐日票池RankIC计算均值、波动和ICIR。"""
        series = pd.Series(values, dtype=float).dropna()
        mean = float(series.mean()) if len(series) else np.nan
        std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        icir = mean / std if np.isfinite(std) and std > 1e-12 else np.nan
        return {"pool_mean_rank_ic": mean, "pool_rank_ic_std": std,
                "pool_icir": float(icir)}

    def run(self, predictions: pd.DataFrame, market: str, output_root: Path, model_name: str,
            retain_signals: pd.DataFrame | None = None, optimizer_name: str = "none"):
        from analytics import performance
        from backtest import BacktestEngine
        from config import BacktestConfig
        from data_loader import MarketData

        wide = self.prediction_wide(predictions)
        # 先让现有加载器按其默认合法信号完成行情/票池加载，随后仅替换内存中的factor。
        cfg = BacktestConfig(market=market, start_date=str(wide.index.min().date()),
                             end_date=str(wide.index.max().date()), signal="default_factor")
        data = MarketData(cfg).load()
        common_dates = data["twap"].index.intersection(wide.index)
        common_codes = data["twap"].columns.intersection(wide.columns)
        for key in ("twap", "close", "adj", "st", "pool"):
            data[key] = data[key].reindex(index=common_dates, columns=common_codes)
        data["factor"] = wide.reindex(index=common_dates, columns=common_codes)
        if retain_signals is not None and not retain_signals.empty:
            retain = retain_signals.copy()
            retain["symbol"] = retain.symbol.astype(str).str.zfill(6)
            retain_wide = retain.pivot_table(index="factor_date", columns="symbol", values="retain",
                                             aggfunc="last").astype(bool)
            data["retain"] = retain_wide.reindex(index=common_dates, columns=common_codes, fill_value=False)
        data["benchmark"] = data["benchmark"].reindex(common_dates)
        root = Path(output_root)
        for name in ("results", "trades", "holdings", "adjustments", "audit"):
            (root / name).mkdir(parents=True, exist_ok=True)
        optimizer = create_optimizer(optimizer_name)
        data["factor"], optimizer_audit = optimizer.transform(data["factor"], data["pool"])
        if not optimizer_audit.empty:
            optimizer_audit.to_csv(
                root / "audit" / f"{model_name}_{market}_{optimizer_name}.csv",
                index=False, encoding="utf-8-sig")
        live_path = root / "results" / f"{model_name}_{market}_live.json"
        live_curve: list[dict[str, Any]] = []
        first_benchmark = float(data["benchmark"].dropna().iloc[0])

        def daily_callback(date, orders, holdings, total_asset, initial_cash):
            benchmark_value = data["benchmark"].get(date, np.nan)
            strategy_value = total_asset / initial_cash
            live_curve.append({"date": str(pd.Timestamp(date).date()),
                               "strategy": float(strategy_value) if np.isfinite(strategy_value) else None,
                               "benchmark": float(benchmark_value / first_benchmark) if pd.notna(benchmark_value) else None,
                               "orders": len(orders), "holdings": len(holdings)})
            payload = {"model": model_name, "pool": market, "current_date": str(pd.Timestamp(date).date()),
                       "completed_days": len(live_curve), "total_days": len(data["twap"]),
                       "curve": live_curve}
            live_path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")

        result = BacktestEngine(data, cfg, daily_callback=daily_callback).run()
        result_dir = root / "results" / f"{model_name}_{market}"
        result_dir.mkdir(parents=True, exist_ok=True)
        result.account.to_csv(result_dir / "account_daily.csv", encoding="utf-8-sig")
        result.orders.to_csv(root / "trades" / f"{model_name}_{market}_orders.csv", index=False, encoding="utf-8-sig")
        result.trades.to_csv(root / "trades" / f"{model_name}_{market}_closed_trades.csv", index=False, encoding="utf-8-sig")
        result.holdings.to_csv(root / "holdings" / f"{model_name}_{market}_holdings_daily.csv", index=False, encoding="utf-8-sig")

        adjustments = []
        if not result.holdings.empty:
            held = result.holdings[["date", "code"]].drop_duplicates()
            for row in held.itertuples(index=False):
                position = data["adj"].index.get_loc(row.date)
                if position > 0:
                    old, new = data["adj"].iloc[position - 1].get(row.code), data["adj"].iloc[position].get(row.code)
                    if pd.notna(old) and pd.notna(new) and old > 0 and not np.isclose(old, new):
                        adjustments.append({"date": row.date, "code": row.code, "old_adj": old,
                                            "new_adj": new, "share_ratio": new / old})
        pd.DataFrame(adjustments, columns=["date", "code", "old_adj", "new_adj", "share_ratio"]).to_csv(
            root / "adjustments" / f"{model_name}_{market}_adjustments.csv", index=False, encoding="utf-8-sig")

        metrics, curve = performance(result.account, result.benchmark, cfg.annual_days, cfg.risk_free_rate)
        pool_rank_ic, pool_ic_stock_count, pool_signal_coverage = [], [], []
        normalized_predictions = predictions.copy()
        normalized_predictions["symbol"] = normalized_predictions.symbol.astype(str).str.zfill(6)
        for date, group in normalized_predictions.groupby("factor_date"):
            date = pd.Timestamp(date)
            if date not in data["pool"].index:
                continue
            members = data["pool"].columns[data["pool"].loc[date].fillna(False).astype(bool)]
            labels = group.drop_duplicates("symbol", keep="last").set_index("symbol")["label"]
            evaluated = pd.DataFrame({
                "prediction": data["factor"].loc[date].reindex(members),
                "label": labels.reindex(members),
            }).dropna()
            pool_ic_stock_count.append(len(evaluated))
            pool_signal_coverage.append(len(evaluated) / len(members) if len(members) else np.nan)
            if len(evaluated) > 1 and evaluated.prediction.std() > 1e-12 and evaluated.label.std() > 1e-12:
                pool_rank_ic.append(evaluated.prediction.corr(evaluated.label, method="spearman"))
        metrics.update(self.pool_ic_statistics(pool_rank_ic))
        metrics.update({"total_fees": float(result.orders.fee.sum()) if not result.orders.empty else 0.0,
                        "turnover": float(result.orders.gross.sum() / result.account.total_asset.mean()) if not result.orders.empty else 0.0,
                        "trade_count": int(len(result.orders)),
                        "pool_rank_ic_positive_ratio": float(np.mean(np.asarray(pool_rank_ic) > 0)) if pool_rank_ic else np.nan,
                        "pool_ic_average_stock_count": float(np.nanmean(pool_ic_stock_count)) if pool_ic_stock_count else np.nan,
                        "pool_signal_coverage": float(np.nanmean(pool_signal_coverage)) if pool_signal_coverage else np.nan,
                        "optimizer": optimizer_name})
        curve.to_csv(result_dir / "nav_curve.csv", encoding="utf-8-sig")
        (result_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        first_order = result.orders.date.min() if not result.orders.empty else pd.NaT
        expected_first_trade = common_dates[1] if len(common_dates) > 1 else pd.NaT
        audit = pd.DataFrame([
            {"check": "account_nonfinite_total_asset", "value": int((~np.isfinite(result.account.total_asset)).sum()), "passed": np.isfinite(result.account.total_asset).all()},
            {"check": "curve_matches_account", "value": bool(np.allclose(curve.strategy, result.account.total_asset / result.account.total_asset.iloc[0])), "passed": bool(np.allclose(curve.strategy, result.account.total_asset / result.account.total_asset.iloc[0]))},
            {"check": "prediction_kept_on_t0", "value": str(wide.index.min()), "passed": wide.index.min() == common_dates.min()},
            {"check": "first_trade_uses_previous_signal", "value": str(first_order), "passed": pd.Timestamp(first_order) == pd.Timestamp(expected_first_trade)},
            {"check": "retain_signal_enabled", "value": retain_signals is not None and not retain_signals.empty, "passed": True},
            {"check": "score_optimizer", "value": optimizer_name, "passed": optimizer_name in {"none", "industry_neutral"}},
            {"check": "industry_neutral_max_abs_mean",
             "value": float(optimizer_audit.max_abs_industry_mean.max()) if not optimizer_audit.empty else np.nan,
             "passed": bool(optimizer_audit.empty or optimizer_audit.max_abs_industry_mean.fillna(0).max() < 1e-10)},
            {"check": "pool_mean_rank_ic", "value": metrics["pool_mean_rank_ic"], "passed": np.isfinite(metrics["pool_mean_rank_ic"])},
            {"check": "max_abs_strategy_daily_return", "value": float(curve.strategy.pct_change(fill_method=None).abs().max()), "passed": True},
            {"check": "max_abs_benchmark_daily_return", "value": float(curve.benchmark.pct_change(fill_method=None).abs().max()), "passed": True},
        ])
        audit.to_csv(root / "audit" / f"{model_name}_{market}_consistency.csv", index=False, encoding="utf-8-sig")
        return result, metrics, curve.reset_index()
