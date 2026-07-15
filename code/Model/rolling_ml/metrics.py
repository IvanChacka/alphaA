from imports import *


def daily_ic(frame: pd.DataFrame, prediction: str = "prediction", label: str = "label") -> pd.DataFrame:
    """按因子日期计算横截面指标。"""
    def one(group: pd.DataFrame) -> pd.Series:
        valid = group[[prediction, label]].dropna()
        prediction_std = valid[prediction].std()
        label_std = valid[label].std()
        if len(valid) < 2 or not np.isfinite(prediction_std) or prediction_std <= 1e-12 or not np.isfinite(label_std) or label_std <= 1e-12:
            return pd.Series({"rank_ic": np.nan, "pearson_ic": np.nan, "stock_count": len(valid),
                              "prediction_std": prediction_std, "label_std": label_std})
        return pd.Series({
            "rank_ic": valid[prediction].corr(valid[label], method="spearman"),
            "pearson_ic": valid[prediction].corr(valid[label]),
            "stock_count": len(valid),
            "prediction_std": prediction_std,
            "label_std": label_std,
        })
    return frame.groupby("factor_date", observed=True).apply(one, include_groups=False).reset_index()


def prediction_turnover(frame: pd.DataFrame) -> float:
    """相邻交易日预测横截面排名的平均绝对变化。"""
    ranked = frame.pivot_table(index="factor_date", columns="symbol", values="prediction")
    ranked = ranked.rank(axis=1, pct=True)
    return float(ranked.diff().abs().mean(axis=1).mean()) if len(ranked) > 1 else np.nan


def metric_summary(frame: pd.DataFrame, annual_days: int = 252) -> dict[str, float]:
    evaluated = frame.dropna(subset=["label", "prediction"])
    daily = daily_ic(evaluated)
    rank_ic = daily.rank_ic.dropna()
    std = rank_ic.std(ddof=1)
    nonannual = rank_ic.mean() / std if std and not np.isnan(std) else np.nan
    ranked = evaluated.copy()
    ranked["bucket"] = ranked.groupby("factor_date").prediction.transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    top = ranked.loc[ranked.bucket == 4, "label"].mean()
    bottom = ranked.loc[ranked.bucket == 0, "label"].mean()
    return {
        "mean_rank_ic": rank_ic.mean(), "rank_ic_std": std, "icir": nonannual,
        "rank_ic_positive_ratio": (rank_ic > 0).mean(),
        "rank_ic_t_stat": (stats.ttest_1samp(rank_ic, 0, nan_policy="omit").statistic
                           if len(rank_ic) > 1 and rank_ic.std(ddof=1) > 0 else np.nan),
        "top_quantile_return": top, "bottom_quantile_return": bottom,
        "top_bottom_return": top - bottom, "prediction_turnover": prediction_turnover(evaluated),
        "observation_days": int(len(rank_ic)), "sample_count": int(len(evaluated)),
    }


def layered_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_data in predictions.groupby("model"):
        rows.append({"model": model, "level": "full_test", "period": "all", **metric_summary(model_data)})
        for year, group in model_data.groupby(model_data.factor_date.dt.year):
            rows.append({"model": model, "level": "year", "period": str(year), **metric_summary(group)})
        for quarter, group in model_data.groupby(model_data.factor_date.dt.to_period("Q")):
            rows.append({"model": model, "level": "quarter", "period": str(quarter), **metric_summary(group)})
    return pd.DataFrame(rows)
