import numpy as np
import pandas as pd


def performance(account: pd.DataFrame, benchmark: pd.Series, annual_days=252, rf=0.0):
    if account.empty:
        raise ValueError("账户记录为空，无法计算绩效")
    invalid = ~np.isfinite(account.total_asset.to_numpy(dtype=float))
    if invalid.any():
        dates = account.index[invalid].astype(str).tolist()[:5]
        raise ValueError(f"账户总资产存在非有限值，示例日期：{dates}")
    nav = account.total_asset / account.total_asset.iloc[0]
    bench = benchmark / benchmark.iloc[0]
    if not np.isfinite(bench.to_numpy(dtype=float)).all():
        raise ValueError("基准净值存在非有限值")
    ret = nav.pct_change(fill_method=None).dropna()
    # Keep every excess metric on the same relative-NAV basis as the chart.
    # Using ``strategy_return - benchmark_return`` and compounding it produces
    # a different path from strategy_nav / benchmark_nav whenever the
    # benchmark return is non-zero.
    excess_nav = nav / bench
    excess = excess_nav.pct_change(fill_method=None).dropna()
    # ``annual_days`` is correct for daily data, but factor-driven backtests can
    # be weekly/monthly. Annualize from the actual date span and infer the number
    # of observations per year for volatility/Sharpe.
    if len(ret) > 1:
        median_days = float(np.median(np.diff(pd.DatetimeIndex(nav.index).asi8) / 86_400_000_000_000))
        periods_per_year = (12.0 if median_days >= 20 else 52.0 if median_days >= 5 else float(annual_days))
    else:
        periods_per_year = float(annual_days)
    elapsed_years = max(len(ret) / periods_per_year, 1 / periods_per_year)
    active_mask = (account["holding_count"].reindex(ret.index).fillna(0).gt(0)
                   if "holding_count" in account else pd.Series(True, index=ret.index))
    def calc(r):
        wealth = r.add(1).cumprod()
        annual_return = float(wealth.iloc[-1] ** (1 / elapsed_years) - 1)
        annual_volatility = float(r.std(ddof=1) * np.sqrt(periods_per_year))
        max_drawdown = float((wealth / wealth.cummax() - 1).min())
        wins, losses = r[r > 0], r[r < 0]
        return {
            "annual_return": annual_return,
            "sharpe": float((annual_return - rf) / annual_volatility) if annual_volatility > 0 else np.nan,
            "annual_volatility": annual_volatility,
            "max_drawdown": max_drawdown,
            "calmar": float(annual_return / abs(max_drawdown)) if max_drawdown < 0 else np.nan,
            "win_rate": float((r > 0).mean()),
            "profit_loss_ratio": (float(wins.mean() / abs(losses.mean()))
                                  if len(wins) and len(losses) else np.nan),
            # Product definition: zero-return dates divided by active holding dates.
            "underwater_ratio": float(r.loc[active_mask.reindex(r.index, fill_value=False)].eq(0).mean())
                                if active_mask.reindex(r.index, fill_value=False).any() else np.nan,
        }
    metrics = {f"absolute_{k}": v for k, v in calc(ret).items()}
    metrics.update({f"excess_{k}": v for k, v in calc(excess).items()})
    curve = pd.DataFrame({"strategy": nav, "benchmark": bench, "excess": excess_nav})
    return metrics, curve
