from imports import *


class TimingDatasetBuilder:
    """在决策日仅使用已于该日收盘前实现的风格收益。"""

    def build_features(self, returns: pd.DataFrame, dates: pd.DatetimeIndex,
                       benchmark: pd.Series | None = None) -> pd.DataFrame:
        styles = sorted(returns["style"].unique())
        rows = []
        for date in dates:
            available = returns[returns.style_exit_date <= date]
            row: dict[str, Any] = {"decision_date": date,
                                   "max_information_date": available.style_exit_date.max() if len(available) else pd.NaT}
            for style in styles:
                values = available.loc[available["style"] == style, "style_return"]
                row[f"{style}_momentum_5"] = values.tail(5).mean()
                row[f"{style}_momentum_20"] = values.tail(20).mean()
                row[f"{style}_volatility_5"] = values.tail(5).std(ddof=0)
            if benchmark is not None:
                benchmark_returns = benchmark.pct_change(fill_method=None).loc[:date].tail(20)
                row["benchmark_volatility_20"] = benchmark_returns.std(ddof=0)
                row["benchmark_return_20"] = ((1 + benchmark_returns.dropna()).prod() - 1
                                               if benchmark_returns.notna().any() else np.nan)
            rows.append(row)
        return pd.DataFrame(rows)

    def build_horizon_labels(self, returns: pd.DataFrame, dates: pd.DatetimeIndex,
                             horizon: int) -> pd.DataFrame:
        """构造指定期限的逐风格复合收益，并记录标签完全实现的日期。"""
        styles = sorted(returns["style"].unique())
        wide = returns.pivot(index="factor_date", columns="style", values="style_return").reindex(dates)
        exits = returns.groupby("factor_date").style_exit_date.max().reindex(dates)
        rows = []
        for i, date in enumerate(dates):
            future = wide.iloc[i:i + horizon]
            if len(future) < horizon or future.isna().any().any():
                continue
            total = (1 + future).prod() - 1
            row = {"decision_date": date, f"label_exit_date_{horizon}": exits.iloc[i + horizon - 1]}
            row.update({f"target_{style}_{horizon}": total[style] for style in styles})
            rows.append(row)
        return pd.DataFrame(rows)

    def build_labels(self, returns: pd.DataFrame, dates: pd.DatetimeIndex,
                     horizon: int = 5) -> pd.DataFrame:
        styles = sorted(returns["style"].unique())
        wide = returns.pivot(index="factor_date", columns="style", values="style_return").reindex(dates)
        exits = returns.groupby("factor_date").style_exit_date.max().reindex(dates)
        rows = []
        for i, date in enumerate(dates):
            future = wide.iloc[i:i + horizon]
            if len(future) < horizon or future.isna().any().any(): continue
            total = (1 + future).prod() - 1
            ordered = total.sort_values(ascending=False)
            row = {"decision_date": date, "best_style": ordered.index[0],
                   "winner_margin": ordered.iloc[0] - ordered.iloc[1],
                   "label_exit_date": exits.iloc[i + horizon - 1]}
            row.update({f"target_{s}": total[s] for s in styles})
            rows.append(row)
        return pd.DataFrame(rows)
