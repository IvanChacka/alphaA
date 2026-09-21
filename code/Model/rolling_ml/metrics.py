from imports import *


def daily_ic(frame: pd.DataFrame, prediction: str = "prediction", label: str = "label") -> pd.DataFrame:
    """按因子日期计算横截面指标。"""
    columns = ["factor_date", "rank_ic", "pearson_ic", "stock_count",
               "prediction_std", "label_std"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
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
    return frame.groupby("factor_date", observed=True).apply(
        one, include_groups=False).reset_index().reindex(columns=columns)


def prediction_turnover(frame: pd.DataFrame) -> float:
    """相邻交易日预测横截面排名的平均绝对变化。"""
    if frame.empty or not {"factor_date", "symbol", "prediction"}.issubset(frame.columns):
        return np.nan
    ranked = frame.pivot_table(index="factor_date", columns="symbol", values="prediction")
    ranked = ranked.rank(axis=1, pct=True)
    return float(ranked.diff().abs().mean(axis=1).mean()) if len(ranked) > 1 else np.nan


def metric_summary(frame: pd.DataFrame, annual_days: int = 252) -> dict[str, float]:
    evaluated = frame.dropna(subset=["label", "prediction"])
    if evaluated.empty:
        return {"mean_rank_ic": np.nan, "pearson_ic": np.nan, "rank_ic_std": np.nan, "icir": np.nan,
                "rank_ic_positive_ratio": np.nan, "rank_ic_t_stat": np.nan,
                "top_quantile_return": np.nan, "bottom_quantile_return": np.nan,
                "top_bottom_return": np.nan, "prediction_turnover": np.nan,
                "observation_days": 0, "sample_count": 0}
    daily = daily_ic(evaluated)
    rank_ic = daily.rank_ic.dropna()
    pearson_ic = daily.pearson_ic.dropna()
    std = rank_ic.std(ddof=1)
    nonannual = rank_ic.mean() / std if std and not np.isnan(std) else np.nan
    ranked = evaluated.copy()
    ranked["bucket"] = ranked.groupby("factor_date").prediction.transform(
        lambda x: pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
    )
    top = ranked.loc[ranked.bucket == 4, "label"].mean()
    bottom = ranked.loc[ranked.bucket == 0, "label"].mean()
    return {
        "mean_rank_ic": rank_ic.mean(), "pearson_ic": pearson_ic.mean(), "rank_ic_std": std, "icir": nonannual,
        "rank_ic_positive_ratio": (rank_ic > 0).mean(),
        "rank_ic_t_stat": (stats.ttest_1samp(rank_ic, 0, nan_policy="omit").statistic
                           if len(rank_ic) > 1 and rank_ic.std(ddof=1) > 0 else np.nan),
        "top_quantile_return": top, "bottom_quantile_return": bottom,
        "top_bottom_return": top - bottom, "prediction_turnover": prediction_turnover(evaluated),
        "observation_days": int(len(rank_ic)), "sample_count": int(len(evaluated)),
    }


def model_selection_score(metrics: dict[str, float]) -> float:
    """联合目标：RankIC、Pearson IC 和 Top-Bottom 收益。

    收益项用 tanh 压缩到 [-1, 1]，避免极端单期收益压过 IC；三项均要求
    有限，调参不会再只追逐单一 RankIC。
    """
    rank_ic = float(metrics.get("mean_rank_ic", np.nan))
    pearson = float(metrics.get("pearson_ic", np.nan))
    top_bottom = float(metrics.get("top_bottom_return", np.nan))
    if not all(np.isfinite(value) for value in (rank_ic, pearson, top_bottom)):
        return np.nan
    return float(.40 * rank_ic + .20 * pearson + .40 * np.tanh(top_bottom * 5.0))


def layered_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model, model_data in predictions.groupby("model"):
        rows.append({"model": model, "level": "full_test", "period": "all", **metric_summary(model_data)})
        for year, group in model_data.groupby(model_data.factor_date.dt.year):
            rows.append({"model": model, "level": "year", "period": str(year), **metric_summary(group)})
        for quarter, group in model_data.groupby(model_data.factor_date.dt.to_period("Q")):
            rows.append({"model": model, "level": "quarter", "period": str(quarter), **metric_summary(group)})
    return pd.DataFrame(rows)
