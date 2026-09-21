"""Portfolio construction, constraints, costs and performance metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


FREQUENCIES = {"D": "daily", "W": "weekly", "M": "monthly", "Q": "quarterly"}


@dataclass(frozen=True)
class PortfolioConfig:
    rebalance_frequency: str = "M"
    top_n: int = 200
    weight_mode: str = "score"
    commission_rate: float = 0.0003
    stamp_duty_rate: float = 0.0005
    max_turnover: float = 1.0
    no_trade_threshold: float = 0.0
    max_drawdown: float | None = None
    industry_neutral: bool = False
    risk_free_rate: float = 0.02

    def validate(self):
        frequency = self.rebalance_frequency.upper()
        if frequency not in FREQUENCIES:
            raise ValueError("调仓频率必须是 D/W/M/Q")
        if self.top_n < 1:
            raise ValueError("Top-N 必须大于 0")
        if self.weight_mode not in {"equal", "score"}:
            raise ValueError("权重方式必须是 equal 或 score")
        if not 0 <= self.max_turnover <= 2:
            raise ValueError("换手率约束必须在 0 到 2 之间")
        if self.max_drawdown is not None and not 0 < self.max_drawdown < 1:
            raise ValueError("最大回撤阈值必须在 0 到 1 之间")
        return self


def performance_metrics(returns: pd.Series, risk_free_rate: float = .02,
                        periods_per_year: int = 252) -> dict[str, float]:
    values = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return {key: np.nan for key in ("annual_return", "annual_volatility", "sharpe",
                "max_drawdown", "calmar", "win_rate", "profit_loss_ratio", "underwater_ratio")}
    nav = values.add(1).cumprod()
    years = max(len(values) / periods_per_year, 1 / periods_per_year)
    annual_return = float(nav.iloc[-1] ** (1 / years) - 1)
    annual_volatility = float(values.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = ((annual_return - risk_free_rate) / annual_volatility
              if annual_volatility > 0 else np.nan)
    drawdown = nav / nav.cummax() - 1
    maximum = float(drawdown.min())
    wins, losses = values[values > 0], values[values < 0]
    profit_loss = (float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else np.nan)
    return {"annual_return": annual_return, "annual_volatility": annual_volatility,
            "sharpe": float(sharpe), "max_drawdown": maximum,
            "calmar": float(annual_return / abs(maximum)) if maximum < 0 else np.nan,
            "win_rate": float((values > 0).mean()), "profit_loss_ratio": profit_loss,
            "underwater_ratio": float(values.eq(0).mean())}


class PortfolioEngine:
    """Long-only Top-N portfolio; a signal at t earns the return from t to t+1."""

    def __init__(self, config: PortfolioConfig):
        self.config = config.validate()

    def _rebalance_dates(self, dates: pd.DatetimeIndex) -> set[pd.Timestamp]:
        frequency = self.config.rebalance_frequency.upper()
        if frequency == "D":
            return set(dates)
        periods = dates.to_period(frequency)
        return set(pd.Series(dates, index=dates).groupby(periods).first())

    def _target(self, scores: pd.Series, industries: pd.Series | None = None) -> pd.Series:
        clean = scores.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        selected = clean.head(self.config.top_n)
        target = pd.Series(0.0, index=scores.index)
        if selected.empty:
            return target
        if self.config.weight_mode == "equal":
            target.loc[selected.index] = 1 / len(selected)
        else:
            ranked = selected.rank(method="average", pct=True)
            target.loc[selected.index] = ranked / ranked.sum()
        if self.config.industry_neutral and industries is not None:
            active = target[target > 0]
            groups = industries.reindex(active.index).fillna("UNKNOWN")
            industry_budget = 1 / max(groups.nunique(), 1)
            for _, names in groups.groupby(groups).groups.items():
                block = target.loc[names]
                target.loc[names] = block / block.sum() * industry_budget
        return target

    def _constrain(self, target: pd.Series, current: pd.Series, drawdown: float) -> pd.Series:
        delta = target - current
        delta[delta.abs() < self.config.no_trade_threshold] = 0
        turnover = float(delta.abs().sum())
        if turnover > self.config.max_turnover > 0:
            target = current + delta * (self.config.max_turnover / turnover)
        if self.config.max_drawdown and drawdown <= -self.config.max_drawdown:
            target *= max(0.0, 1 + drawdown / self.config.max_drawdown)
        return target.clip(lower=0)

    def run(self, scores: pd.DataFrame, forward_returns: pd.DataFrame,
            industries: pd.Series | None = None) -> dict[str, Any]:
        dates = scores.index.intersection(forward_returns.index).sort_values()
        symbols = scores.columns.intersection(forward_returns.columns)
        if len(dates) == 0 or len(symbols) == 0:
            raise ValueError("得分与收益没有共同日期或股票")
        scores, returns = scores.reindex(index=dates, columns=symbols), forward_returns.reindex(index=dates, columns=symbols)
        rebalances = self._rebalance_dates(dates)
        weights = pd.Series(0.0, index=symbols)
        nav = 1.0
        peak = 1.0
        rows = []
        for date in dates:
            turnover = cost = 0.0
            if date in rebalances:
                target = self._target(scores.loc[date], industries)
                target = self._constrain(target, weights, nav / peak - 1)
                delta = target - weights
                buys = float(delta.clip(lower=0).sum())
                sells = float((-delta.clip(upper=0)).sum())
                turnover = buys + sells
                cost = (buys + sells) * self.config.commission_rate + sells * self.config.stamp_duty_rate
                weights = target
            gross = float((weights * returns.loc[date].fillna(0)).sum())
            net = gross - cost
            nav *= 1 + net
            peak = max(peak, nav)
            rows.append({"date": date, "gross_return": gross, "cost": cost,
                         "net_return": net, "turnover": turnover, "nav": nav,
                         "holding_count": int((weights > 0).sum())})
            drift = weights * (1 + returns.loc[date].fillna(0))
            weights = drift / drift.sum() if drift.sum() > 0 else drift
        curve = pd.DataFrame(rows).set_index("date")
        return {"curve": curve, "metrics": performance_metrics(
            curve.net_return, self.config.risk_free_rate),
            "total_cost": float(curve.cost.sum()), "average_turnover": float(curve.turnover.mean())}
