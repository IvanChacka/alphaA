"""IC, layer and factor research services independent of model training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HORIZONS = (5, 10, 20, 60)


class ICAnalyzer:
    def __init__(self, horizons: tuple[int, ...] = HORIZONS):
        self.horizons = tuple(int(value) for value in horizons)

    @staticmethod
    def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
        # Decision at t is executed at t+1, then held for ``horizon`` sessions.
        return prices.shift(-(horizon + 1)).div(prices.shift(-1)).sub(1)

    @staticmethod
    def returns_for_signal_dates(prices: pd.DataFrame, dates: pd.DatetimeIndex,
                                 horizon: int) -> pd.DataFrame:
        dates = pd.DatetimeIndex(dates).sort_values().unique()
        entry_positions = prices.index.searchsorted(dates, side="right")
        exit_positions = entry_positions + int(horizon)
        valid = (entry_positions < len(prices)) & (exit_positions < len(prices))
        output = pd.DataFrame(np.nan, index=dates, columns=prices.columns, dtype=float)
        if valid.any():
            entries = prices.iloc[entry_positions[valid]].set_axis(dates[valid])
            exits = prices.iloc[exit_positions[valid]].set_axis(dates[valid])
            output.loc[dates[valid]] = exits.div(entries).sub(1)
        return output

    def analyze(self, factor: pd.DataFrame, prices: pd.DataFrame) -> dict[str, Any]:
        dates = factor.index[(factor.index >= prices.index.min()) & (factor.index <= prices.index.max())]
        symbols = factor.columns.intersection(prices.columns)
        factor = factor.reindex(index=dates, columns=symbols)
        prices = prices.reindex(columns=symbols)
        series_rows, summary_rows = [], []
        for horizon in self.horizons:
            future = self.returns_for_signal_dates(prices, dates, horizon).reindex(columns=symbols)
            rank_values, pearson_values = [], []
            for date in dates:
                pair = pd.concat([factor.loc[date].rename("factor"), future.loc[date].rename("return")], axis=1).dropna()
                if len(pair) < 3 or pair.factor.nunique() < 2 or pair["return"].nunique() < 2:
                    rank_ic = pearson_ic = np.nan
                else:
                    rank_ic = pair.factor.corr(pair["return"], method="spearman")
                    pearson_ic = pair.factor.corr(pair["return"], method="pearson")
                series_rows.append({"date": str(pd.Timestamp(date).date()), "horizon": horizon,
                                    "rank_ic": rank_ic, "pearson_ic": pearson_ic,
                                    "stock_count": len(pair)})
                rank_values.append(rank_ic); pearson_values.append(pearson_ic)
            for method, values in (("spearman", rank_values), ("pearson", pearson_values)):
                clean = pd.Series(values, dtype=float).dropna()
                std = clean.std(ddof=1)
                summary_rows.append({"horizon": horizon, "method": method,
                    "mean": clean.mean(), "std": std,
                    "icir": clean.mean() / std if std and np.isfinite(std) else np.nan,
                    "win_rate": (clean > 0).mean(), "absolute_mean": clean.abs().mean(),
                    "observations": len(clean)})
        return {"summary": summary_rows, "series": series_rows}


class LayeredBacktester:
    def __init__(self, layers: int = 10, holding_period: str = "M"):
        if layers not in {5, 10}:
            raise ValueError("分层数量必须是 5 或 10")
        if holding_period.upper() not in {"D", "W", "M", "Q"}:
            raise ValueError("持有期必须是 D/W/M/Q")
        self.layers, self.holding_period = layers, holding_period.upper()

    def run(self, factor: pd.DataFrame, prices: pd.DataFrame) -> dict[str, Any]:
        frequency_horizon = {"D": 1, "W": 5, "M": 20, "Q": 60}
        dates = factor.index[(factor.index >= prices.index.min()) & (factor.index <= prices.index.max())]
        future = ICAnalyzer.returns_for_signal_dates(
            prices, dates, frequency_horizon[self.holding_period])
        rows = []
        for date in dates:
            pair = pd.concat([factor.loc[date].rename("factor"), future.loc[date].rename("return")], axis=1).dropna()
            if len(pair) < self.layers:
                continue
            pair["layer"] = pd.qcut(pair.factor.rank(method="first"), self.layers,
                                    labels=False, duplicates="drop") + 1
            for layer, group in pair.groupby("layer"):
                rows.append({"date": date, "layer": f"Q{int(layer)}", "return": group["return"].mean()})
        frame = pd.DataFrame(rows)
        if frame.empty:
            return {"curve": [], "long_short": [], "summary": []}
        matrix = frame.pivot(index="date", columns="layer", values="return").sort_index()
        nav = matrix.fillna(0).add(1).cumprod()
        low, high = "Q1", f"Q{self.layers}"
        long_short = matrix.get(high, pd.Series(index=matrix.index, dtype=float)).sub(
            matrix.get(low, pd.Series(index=matrix.index, dtype=float))).fillna(0).add(1).cumprod()
        curve = [{"date": str(date.date()), **{column: value for column, value in row.items()}}
                 for date, row in nav.iterrows()]
        return {"curve": curve,
                "long_short": [{"date": str(date.date()), "nav": value} for date, value in long_short.items()],
                "summary": [{"layer": column, "mean_return": matrix[column].mean(),
                             "win_rate": (matrix[column] > 0).mean()} for column in matrix.columns]}


@dataclass
class FactorResearchEngine:
    horizons: tuple[int, ...] = HORIZONS

    def run(self, factor: pd.DataFrame, prices: pd.DataFrame,
            holding_period: str = "M") -> dict[str, Any]:
        return {"ic": ICAnalyzer(self.horizons).analyze(factor, prices),
                "deciles": LayeredBacktester(10, holding_period).run(factor, prices),
                "quintiles": LayeredBacktester(5, holding_period).run(factor, prices)}
