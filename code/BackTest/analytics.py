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
    ret, bret = nav.pct_change(fill_method=None).dropna(), bench.pct_change(fill_method=None).dropna()
    excess = ret.sub(bret, fill_value=0)
    years = max(len(ret) / annual_days, 1 / annual_days)
    calc = lambda r: {
        "annual_return": float((1 + r).prod() ** (1 / years) - 1),
        "sharpe": float((r.mean() * annual_days - rf) / (r.std(ddof=1) * np.sqrt(annual_days))) if r.std(ddof=1) > 0 else np.nan,
        "annual_volatility": float(r.std(ddof=1) * np.sqrt(annual_days)),
        "max_drawdown": float((r.add(1).cumprod() / r.add(1).cumprod().cummax() - 1).min()),
    }
    metrics = {f"absolute_{k}": v for k, v in calc(ret).items()}
    metrics.update({f"excess_{k}": v for k, v in calc(excess).items()})
    curve = pd.DataFrame({"strategy": nav, "benchmark": bench, "excess": nav / bench})
    return metrics, curve
