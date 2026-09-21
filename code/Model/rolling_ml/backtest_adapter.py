from imports import *
from env import BACKTEST_DIR
from optimizers import (create_drawdown_optimizer, create_optimizer,
                        create_turnover_optimizer)


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
        value = "trade_prediction" if "trade_prediction" in frame else "prediction"
        return frame.pivot_table(index="factor_date", columns="symbol", values=value,
                                 aggfunc="last", dropna=False).sort_index()

    @staticmethod
    def pool_ic_statistics(values: list[float]) -> dict[str, float]:
        """从逐日票池RankIC计算均值、波动和ICIR。"""
        series = pd.Series(values, dtype=float).dropna()
        mean = float(series.mean()) if len(series) else np.nan
        std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
        icir = mean / std if np.isfinite(std) and std > 1e-12 else np.nan
        return {"pool_mean_rank_ic": mean, "pool_rank_ic_std": std,
                "pool_icir": float(icir)}

    @staticmethod
    def score_layer_curve(predictions: pd.DataFrame, layers: int = 10) -> pd.DataFrame:
        """Build score-ranked layer NAVs from out-of-sample model scores."""
        rows = []
        for date, group in predictions.groupby("factor_date", sort=True):
            valid = group.dropna(subset=["prediction", "label"]).copy()
            if len(valid) < layers:
                continue
            valid["bucket"] = pd.qcut(valid["prediction"].rank(method="first"),
                                       layers, labels=False)
            returns = valid.groupby("bucket", observed=True)["label"].mean()
            row = {"date": pd.Timestamp(date)}
            for bucket in range(layers):
                row[f"q{bucket + 1}"] = float(returns.get(bucket, np.nan))
            row["q10_q1"] = row["q10"] - row["q1"]
            rows.append(row)
        columns = ["date", *[f"q{i}" for i in range(1, layers + 1)], "q10_q1"]
        if not rows:
            return pd.DataFrame(columns=columns)
        result = pd.DataFrame(rows).sort_values("date")
        for column in columns[1:]:
            result[column] = (1.0 + result[column].fillna(0.0)).cumprod()
        return result

    def run(self, predictions: pd.DataFrame, market: str, output_root: Path, model_name: str,
            optimizer_name: str = "none", turnover_optimizer_name: str = "none",
            drawdown_optimizer_name: str = "none", sell_confirmations: int = 2,
            max_turnover_ratio: float = .30, max_drawdown_limit: float = .20,
            holding_count: int = 200, selection_mode: str = "top_quantile",
            rotation_quantile: float = .10, rebalance_frequency: str = "daily",
            weight_mode: str = "score", include_costs: bool = False,
            commission_rate: float = .0003, stamp_duty_rate: float = .0005):
        from analytics import performance
        from backtest import BacktestEngine
        from config import BacktestConfig
        from data_loader import MarketData

        wide = self.prediction_wide(predictions)
        # 先让现有加载器按其默认合法信号完成行情/票池加载，随后仅替换内存中的factor。
        turnover_optimizer = create_turnover_optimizer(
            turnover_optimizer_name, sell_confirmations, max_turnover_ratio)
        drawdown_optimizer = create_drawdown_optimizer(
            drawdown_optimizer_name, max_drawdown_limit)
        cfg = BacktestConfig(
            market=market, start_date=str(wide.index.min().date()),
            end_date=str(wide.index.max().date()), signal="default_factor",
            holding_count=int(holding_count),
            weight_mode=str(weight_mode),
            selection_mode=selection_mode,
            rotation_quantile=float(rotation_quantile),
            rebalance_frequency=rebalance_frequency,
            max_turnover_ratio=(turnover_optimizer.max_turnover_ratio
                                if turnover_optimizer.max_turnover_ratio is not None else 1.0),
            max_drawdown_limit=drawdown_optimizer.max_drawdown_limit,
            allow_fractional_shares=market == "ALL_MARKET",
            enforce_price_limits=market != "ALL_MARKET",
            liquidate_missing_at_last_close=market == "ALL_MARKET",
            fee_rate=None if include_costs else 0.0,
            commission_rate=float(commission_rate),
            stamp_duty_rate_before_20230828=float(stamp_duty_rate),
            stamp_duty_rate_from_20230828=float(stamp_duty_rate))
        # ALL_MARKET must intersect prices with this run's predictions. Falling back to
        # default_factor here would silently truncate an uploaded factor's newer dates.
        data = MarketData(cfg, factor_override=wide).load()
        common_dates = data["twap"].index
        common_codes = data["twap"].columns.intersection(wide.columns)
        for key in ("twap", "close", "adj", "st", "pool"):
            data[key] = data[key].reindex(index=common_dates, columns=common_codes)
        # MarketData has already aligned calendar month-end signals (including
        # weekend month ends) to the latest tradable date. Do not overwrite the
        # aligned frame with the original sparse index here.
        data["factor"] = data["factor"].reindex(index=common_dates, columns=common_codes)
        normalized_predictions = predictions.copy()
        normalized_predictions["factor_date"] = pd.to_datetime(normalized_predictions.factor_date)
        normalized_predictions["symbol"] = normalized_predictions.symbol.astype(str).str.zfill(6)
        retain_wide, turnover_audit = turnover_optimizer.build_retain(
            normalized_predictions, common_dates, common_codes)
        if retain_wide is not None:
            data["retain"] = retain_wide
        if "trade_enabled" in normalized_predictions:
            gate = normalized_predictions.groupby("factor_date").trade_enabled.last()
            data["trade_enabled"] = gate.reindex(common_dates, fill_value=False).astype(bool)
        trade_enabled, drawdown_audit = drawdown_optimizer.build_trade_gate(
            normalized_predictions, common_dates)
        if trade_enabled is not None:
            existing_gate = data.get("trade_enabled", pd.Series(True, index=common_dates))
            data["trade_enabled"] = existing_gate & trade_enabled
        if getattr(drawdown_optimizer, "name", "none") != "none":
            data["drawdown_optimizer"] = drawdown_optimizer
            data["risk_off"] = drawdown_audit.set_index("factor_date")["risk_off"].reindex(
                common_dates, fill_value=False).astype(bool)
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
        if not turnover_audit.empty:
            turnover_audit.to_csv(
                root / "audit" / f"{model_name}_{market}_{turnover_optimizer_name}.csv",
                index=False, encoding="utf-8-sig")
        if not drawdown_audit.empty:
            drawdown_audit.to_csv(
                root / "audit" / f"{model_name}_{market}_{drawdown_optimizer_name}.csv",
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
        no_fee_result = BacktestEngine(data, replace(cfg, fee_rate=0.0)).run()
        layer_curve = self.score_layer_curve(normalized_predictions)
        result_dir = root / "results" / f"{model_name}_{market}"
        result_dir.mkdir(parents=True, exist_ok=True)
        result.account.to_csv(result_dir / "account_daily.csv", encoding="utf-8-sig")
        result.orders.to_csv(root / "trades" / f"{model_name}_{market}_orders.csv", index=False, encoding="utf-8-sig")
        result.trades.to_csv(root / "trades" / f"{model_name}_{market}_closed_trades.csv", index=False, encoding="utf-8-sig")
        result.holdings.to_csv(root / "holdings" / f"{model_name}_{market}_holdings_daily.csv", index=False, encoding="utf-8-sig")
        result.risk_events.to_csv(root / "audit" / f"{model_name}_{market}_risk_events.csv",
                                  index=False, encoding="utf-8-sig")

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
        no_fee_metrics, no_fee_curve = performance(
            no_fee_result.account, no_fee_result.benchmark, cfg.annual_days, cfg.risk_free_rate)
        metrics.update({f"no_fee_{key}": value for key, value in no_fee_metrics.items()})
        metrics.update({
            "commission_rate": cfg.commission_rate,
            "transfer_fee_rate": cfg.transfer_fee_rate,
            "minimum_commission": cfg.minimum_commission,
            "stamp_duty_rate_before_20230828": cfg.stamp_duty_rate_before_20230828,
            "stamp_duty_rate_from_20230828": cfg.stamp_duty_rate_from_20230828,
            "no_fee_terminal_asset": float(no_fee_result.account.total_asset.iloc[-1]),
            "net_terminal_asset": float(result.account.total_asset.iloc[-1]),
            "no_fee_sharpe_minus_net": float(
                no_fee_metrics["absolute_sharpe"] - metrics["absolute_sharpe"]),
            "no_fee_excess_sharpe_minus_net": float(
                no_fee_metrics["excess_sharpe"] - metrics["excess_sharpe"]),
        })
        pool_rank_ic, pool_ic_stock_count, pool_signal_coverage = [], [], []
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
                        "optimizer": optimizer_name,
                        "turnover_optimizer": turnover_optimizer_name,
                        "include_costs": bool(include_costs),
                        "backtest_stage": ("optimized" if any(name != "none" for name in
                                               (optimizer_name, turnover_optimizer_name,
                                                drawdown_optimizer_name)) else
                                           "cost" if include_costs else "pure"),
                        "selection_mode": selection_mode,
                        "rotation_quantile": float(rotation_quantile),
                        "rebalance_frequency": rebalance_frequency,
                        "scheduled_rebalance_days": int(result.risk_events.scheduled_rebalance.sum()),
                        "drawdown_optimizer": drawdown_optimizer_name,
                        "risk_paused_days": int((~result.risk_events.trade_enabled).sum()),
                        "qp_adjustment_days": int((result.risk_events.target_exposure < .999).sum()),
                        "average_target_exposure": float(result.risk_events.target_exposure.mean()),
                        "minimum_target_exposure": float(result.risk_events.target_exposure.min())})
        curve.to_csv(result_dir / "nav_curve.csv", encoding="utf-8-sig")
        layer_curve.to_csv(root / "results" / f"{model_name}_{market}_layers.csv",
                           index=False, encoding="utf-8-sig")
        no_fee_curve.to_csv(result_dir / "no_fee_nav_curve.csv", encoding="utf-8-sig")
        (result_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        first_order = result.orders.date.min() if not result.orders.empty else pd.NaT
        expected_first_trade = common_dates[1] if len(common_dates) > 1 else pd.NaT
        audit = pd.DataFrame([
            {"check": "account_nonfinite_total_asset", "value": int((~np.isfinite(result.account.total_asset)).sum()), "passed": np.isfinite(result.account.total_asset).all()},
            {"check": "curve_matches_account", "value": bool(np.allclose(curve.strategy, result.account.total_asset / result.account.total_asset.iloc[0])), "passed": bool(np.allclose(curve.strategy, result.account.total_asset / result.account.total_asset.iloc[0]))},
            {"check": "prediction_kept_on_t0", "value": str(wide.index.min()), "passed": wide.index.min() == common_dates.min()},
            {"check": "first_trade_uses_previous_signal", "value": str(first_order), "passed": pd.Timestamp(first_order) == pd.Timestamp(expected_first_trade)},
            {"check": "turnover_optimizer", "value": turnover_optimizer_name, "passed": True},
            {"check": "drawdown_optimizer", "value": drawdown_optimizer_name, "passed": True},
            {"check": "score_optimizer", "value": optimizer_name, "passed": optimizer_name in {"none", "industry_neutral"}},
            {"check": "industry_neutral_max_abs_mean",
             "value": float(optimizer_audit.max_abs_industry_mean.max()) if not optimizer_audit.empty else np.nan,
             "passed": bool(optimizer_audit.empty or optimizer_audit.max_abs_industry_mean.fillna(0).max() < 1e-10)},
            {"check": "pool_mean_rank_ic", "value": metrics["pool_mean_rank_ic"], "passed": np.isfinite(metrics["pool_mean_rank_ic"])},
            {"check": "max_abs_strategy_daily_return", "value": float(curve.strategy.pct_change(fill_method=None).abs().max()), "passed": True},
            {"check": "max_abs_benchmark_daily_return", "value": float(curve.benchmark.pct_change(fill_method=None).abs().max()), "passed": True},
            {"check": "no_fee_terminal_asset_not_below_net",
             "value": float(no_fee_result.account.total_asset.iloc[-1] - result.account.total_asset.iloc[-1]),
             "passed": bool(no_fee_result.account.total_asset.iloc[-1] >= result.account.total_asset.iloc[-1])},
            {"check": "no_fee_and_net_signal_dates_match",
             "value": len(no_fee_result.account),
             "passed": bool(no_fee_result.account.index.equals(result.account.index))},
        ])
        audit.to_csv(root / "audit" / f"{model_name}_{market}_consistency.csv", index=False, encoding="utf-8-sig")
        return result, metrics, curve.reset_index()
