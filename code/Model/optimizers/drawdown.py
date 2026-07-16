from imports import *
from env import (DRAWDOWN_QP_LOOKBACK, DRAWDOWN_QP_MAX_EXPOSURE_STEP,
                 DRAWDOWN_QP_MIN_EXPOSURE, DRAWDOWN_QP_MIN_OBSERVATIONS,
                 DRAWDOWN_QP_RETURN_REWARD, DRAWDOWN_QP_RISK_AVERSION,
                 DRAWDOWN_QP_TURNOVER_PENALTY)


class NoDrawdownOptimizer:
    name = "none"
    max_drawdown_limit: float | None = None

    def build_trade_gate(self, predictions: pd.DataFrame, dates: pd.DatetimeIndex
                         ) -> tuple[pd.Series | None, pd.DataFrame]:
        return None, pd.DataFrame()


class QuadraticDrawdownOptimizer:
    """Convex quadratic optimizer for the portfolio's total risky exposure.

    The stock-selection layer remains Top-N and equal budget per stock.  This
    optimizer only chooses a scalar exposure in ``[min_exposure, 1]`` using
    information available before the current trade:

        min_e  risk * downside_second_moment * e^2
             + turnover_penalty * (e - previous_e)^2
             - 2 * return_reward * max(trailing_mean, 0) * e

    Current drawdown and the model ``risk_off`` flag increase the risk term.
    The problem is a one-dimensional positive-semidefinite QP, so its exact
    solution is obtained analytically and then projected onto box/step limits.
    This minimizes a drawdown-risk surrogate; it cannot guarantee the realized
    pathwise maximum drawdown.
    """

    name = "quadratic_drawdown"
    max_drawdown_limit: float | None = None

    def __init__(self, drawdown_trigger: float = .20,
                 lookback: int = DRAWDOWN_QP_LOOKBACK,
                 min_observations: int = DRAWDOWN_QP_MIN_OBSERVATIONS,
                 min_exposure: float = DRAWDOWN_QP_MIN_EXPOSURE,
                 risk_aversion: float = DRAWDOWN_QP_RISK_AVERSION,
                 turnover_penalty: float = DRAWDOWN_QP_TURNOVER_PENALTY,
                 return_reward: float = DRAWDOWN_QP_RETURN_REWARD,
                 max_exposure_step: float = DRAWDOWN_QP_MAX_EXPOSURE_STEP):
        if not 0 < drawdown_trigger < 1:
            raise ValueError("回撤风险加速阈值必须在0到1之间")
        if lookback < 2 or not 2 <= min_observations <= lookback:
            raise ValueError("二次规划观察窗口参数无效")
        if not 0 <= min_exposure <= 1:
            raise ValueError("最低风险仓位必须在0到1之间")
        if risk_aversion <= 0 or turnover_penalty <= 0 or return_reward < 0:
            raise ValueError("二次规划惩罚参数必须为正")
        if not 0 < max_exposure_step <= 1:
            raise ValueError("单日仓位变化上限必须在0到1之间")
        self.drawdown_trigger = float(drawdown_trigger)
        self.lookback = int(lookback)
        self.min_observations = int(min_observations)
        self.min_exposure = float(min_exposure)
        self.risk_aversion = float(risk_aversion)
        self.turnover_penalty = float(turnover_penalty)
        self.return_reward = float(return_reward)
        self.max_exposure_step = float(max_exposure_step)

    def build_trade_gate(self, predictions: pd.DataFrame, dates: pd.DatetimeIndex
                         ) -> tuple[pd.Series, pd.DataFrame]:
        """Keep trading enabled and expose the model risk flag for the live QP."""
        risk_off = pd.Series(False, index=dates, dtype=bool)
        if "risk_off" in predictions:
            frame = predictions[["factor_date", "risk_off"]].copy()
            frame["factor_date"] = pd.to_datetime(frame.factor_date)
            conflicts = frame.groupby("factor_date").risk_off.nunique(dropna=False)
            if conflicts.gt(1).any():
                raise ValueError("同一交易日存在冲突的risk_off信号")
            signal = frame.groupby("factor_date").risk_off.last().fillna(False).astype(bool)
            risk_off = signal.reindex(dates, fill_value=False)
        enabled = pd.Series(True, index=dates, name="trade_enabled")
        audit = pd.DataFrame({"factor_date": dates,
                              "risk_off": risk_off.to_numpy(),
                              "trade_enabled": True,
                              "drawdown_trigger": self.drawdown_trigger,
                              "method": self.name})
        return enabled, audit

    def optimize_exposure(self, current_drawdown: float, recent_returns,
                          previous_exposure: float = 1.0,
                          risk_off: bool = False) -> dict[str, float | str | bool | int]:
        values = pd.Series(recent_returns, dtype=float).replace(
            [np.inf, -np.inf], np.nan).dropna().tail(self.lookback)
        previous = float(np.clip(previous_exposure, self.min_exposure, 1.0))
        if len(values) < self.min_observations:
            return {"target_exposure": previous, "unconstrained_exposure": previous,
                    "trailing_mean": np.nan, "downside_second_moment": np.nan,
                    "risk_multiplier": 1.0, "qp_observations": int(len(values)),
                    "qp_status": "warmup", "risk_off": bool(risk_off)}

        trailing_mean = float(values.mean())
        downside = np.minimum(values.to_numpy(dtype=float), 0.0)
        downside_second = float(np.mean(np.square(downside)))
        drawdown_ratio = max(-float(current_drawdown), 0.0) / self.drawdown_trigger
        risk_multiplier = 1.0 + 4.0 * drawdown_ratio ** 2 + (2.0 if risk_off else 0.0)
        quadratic = self.risk_aversion * risk_multiplier * max(downside_second, 1e-12)
        quadratic += self.turnover_penalty
        linear_reward = self.return_reward * max(trailing_mean, 0.0)
        linear_reward += self.turnover_penalty * previous
        unconstrained = linear_reward / quadratic

        lower_step = max(self.min_exposure, previous - self.max_exposure_step)
        upper_step = min(1.0, previous + self.max_exposure_step)
        target = float(np.clip(unconstrained, lower_step, upper_step))
        return {"target_exposure": target,
                "unconstrained_exposure": float(unconstrained),
                "trailing_mean": trailing_mean,
                "downside_second_moment": downside_second,
                "risk_multiplier": float(risk_multiplier),
                "qp_observations": int(len(values)),
                "qp_status": "optimal",
                "risk_off": bool(risk_off)}


# Backward-compatible import and CLI alias. Its behavior is now quadratic,
# not the old hard trading freeze.
MaxDrawdownOptimizer = QuadraticDrawdownOptimizer
