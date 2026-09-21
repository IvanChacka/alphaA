"""A 股多因子回测核心。"""
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd
from config import BacktestConfig


@dataclass
class Position:
    shares: float
    avg_cost: float
    opened: pd.Timestamp


class BacktestCancelled(RuntimeError):
    """用户主动终止回测。"""


class BacktestEngine:
    def __init__(self, data: dict, cfg: BacktestConfig, progress_callback=None,
                 daily_callback=None, cancel_check=None):
        self.d, self.cfg = data, cfg
        self.progress_callback = progress_callback
        self.daily_callback = daily_callback
        self.cancel_check = cancel_check
        self.cash = cfg.initial_cash
        self.peak_asset = cfg.initial_cash
        self.target_exposure = 1.0
        self.positions: dict[str, Position] = {}
        # 停牌或退市期间用最后一个有效收盘价估值；只影响账面估值，不允许交易。
        self.last_close: dict[str, float] = {}
        self.orders, self.closed, self.daily, self.holdings, self.risk_events = [], [], [], [], []

    @staticmethod
    def lot(code: str) -> int:
        return 200 if code.startswith("688") else 100

    @staticmethod
    def actual(adjusted: float, adj: float) -> float:
        return adjusted / adj if pd.notna(adjusted) and pd.notna(adj) and adj > 0 else np.nan

    def _tradeable(self, date, code, side, prev_close) -> bool:
        twap = self.actual(self.d["twap"].at[date, code], self.d["adj"].at[date, code])
        if not np.isfinite(twap):
            return False
        if side == "buy" and bool(self.d["st"].at[date, code]):
            return False
        yesterday = prev_close.get(code, np.nan)
        if self.cfg.enforce_price_limits and np.isfinite(yesterday) and yesterday > 0:
            limit = .20 if code.startswith(("300", "688")) else .10
            change = twap / yesterday - 1
            if change >= limit - 1e-6 or change <= -limit + 1e-6:
                return False
        return True

    def _transaction_fee(self, date, side: str, gross: float) -> float:
        if self.cfg.fee_rate is not None:
            return gross * self.cfg.fee_rate
        commission = (max(gross * self.cfg.commission_rate, self.cfg.minimum_commission)
                      if self.cfg.commission_rate > 0 else 0.0)
        transfer = gross * self.cfg.transfer_fee_rate
        stamp = 0.0
        if side == "SELL":
            rate = (self.cfg.stamp_duty_rate_from_20230828
                    if pd.Timestamp(date) >= pd.Timestamp("2023-08-28")
                    else self.cfg.stamp_duty_rate_before_20230828)
            stamp = gross * rate
        return float(commission + transfer + stamp)

    def _sell_shares(self, date, code, shares, prev_close):
        p = self.positions[code]
        current_price = self.actual(
            self.d["twap"].at[date, code], self.d["adj"].at[date, code])
        if self._tradeable(date, code, "sell", prev_close):
            price = current_price
        elif self.cfg.liquidate_missing_at_last_close and not np.isfinite(current_price):
            price = self.last_close.get(code, prev_close.get(code, p.avg_cost))
            if not np.isfinite(price) or price <= 0:
                return False
        else:
            return False
        if shares < p.shares and not self.cfg.allow_fractional_shares:
            lot = self.lot(code)
            shares = math.floor(shares / lot) * lot
        shares = min(float(shares), p.shares)
        if shares <= 0:
            return False
        gross = shares * price
        fee = self._transaction_fee(date, "SELL", gross)
        self.cash += gross - fee
        pnl = (price - p.avg_cost) * shares - fee
        self.orders.append([date, code, "SELL", shares, price, fee, gross])
        self.closed.append([code, p.opened, date, p.avg_cost, price, shares, pnl,
                            (date - p.opened).days])
        p.shares -= shares
        if p.shares <= 1e-9:
            del self.positions[code]
            self.last_close.pop(code, None)
        return True

    def _sell(self, date, code, prev_close):
        return self._sell_shares(date, code, self.positions[code].shares, prev_close)

    def _buy(self, date, code, budget, prev_close):
        if not self._tradeable(date, code, "buy", prev_close): return False
        price = self.actual(self.d["twap"].at[date, code], self.d["adj"].at[date, code])
        lot = self.lot(code)
        available = min(budget, self.cash)
        variable_rate = (self.cfg.fee_rate if self.cfg.fee_rate is not None
                         else self.cfg.commission_rate + self.cfg.transfer_fee_rate)
        if self.cfg.allow_fractional_shares:
            shares = available / (price * (1 + variable_rate))
        else:
            shares = math.floor(available / (price * (1 + variable_rate)) / lot) * lot
        if shares <= 0: return False
        gross = shares * price
        fee = self._transaction_fee(date, "BUY", gross)
        while shares > 0 and gross + fee > available:
            if self.cfg.allow_fractional_shares:
                shares *= available / (gross + fee)
            else:
                shares -= lot
            gross = shares * price
            fee = self._transaction_fee(date, "BUY", gross) if shares > 0 else 0.0
        if shares <= 0:
            return False
        self.cash -= gross + fee
        unit_cost = price + fee / shares
        if code in self.positions:
            position = self.positions[code]
            total_shares = position.shares + shares
            position.avg_cost = (position.avg_cost * position.shares + unit_cost * shares) / total_shares
            position.shares = total_shares
        else:
            self.positions[code] = Position(shares, unit_cost, date)
        self.orders.append([date, code, "BUY", shares, price, fee, gross])
        return True

    def _rebalance_exposure(self, date, target_exposure, equity, prev_close):
        """Scale existing positions proportionally while keeping stock selection unchanged."""
        if not self.positions:
            return
        desired_per_stock = equity * target_exposure / max(self.cfg.holding_count, 1)
        # Sell first so purchases never depend on unavailable cash.
        for code in list(self.positions):
            price = self.actual(self.d["twap"].at[date, code], self.d["adj"].at[date, code])
            if not np.isfinite(price):
                continue
            current = self.positions[code].shares * price
            excess = current - desired_per_stock
            if excess >= price * self.lot(code):
                self._sell_shares(date, code, excess / price, prev_close)
        for code in list(self.positions):
            price = self.actual(self.d["twap"].at[date, code], self.d["adj"].at[date, code])
            if not np.isfinite(price):
                continue
            current = self.positions[code].shares * price
            shortage = desired_per_stock - current
            if shortage >= price * self.lot(code):
                self._buy(date, code, shortage, prev_close)

    def _allocation_weights(self, scores, codes):
        codes = list(codes)
        if not codes:
            return {}
        if getattr(self.cfg, "weight_mode", "equal") == "equal":
            value = 1.0 / len(codes)
            return {code: value for code in codes}
        values = pd.to_numeric(scores.reindex(codes), errors="coerce")
        # Model outputs are ranking signals, not calibrated expected returns.
        # Weight by cross-sectional score rank so a single scale outlier cannot
        # absorb most of the portfolio while higher scores still receive more.
        values = values.rank(method="average", pct=True).fillna(0.0)
        total = float(values.sum())
        if not np.isfinite(total) or total <= 1e-12:
            value = 1.0 / len(codes)
            return {code: value for code in codes}
        # Keep every selected stock investable while letting the model score tilt size.
        score_weight = values / total
        base_weight = 0.20 / len(codes)
        return {code: float(base_weight + 0.80 * score_weight.loc[code]) for code in codes}

    def _corporate_actions(self, date, prev_date):
        for code, p in self.positions.items():
            old, new = self.d["adj"].at[prev_date, code], self.d["adj"].at[date, code]
            if pd.notna(old) and pd.notna(new) and old > 0 and not np.isclose(old, new):
                ratio = new / old
                p.shares *= ratio
                p.avg_cost /= ratio
                if code in self.last_close:
                    self.last_close[code] /= ratio

    def _close_prices(self, date) -> pd.Series:
        return self.d["close"].loc[date] / self.d["adj"].loc[date]

    def _scheduled_rebalance(self, position: int, dates: pd.DatetimeIndex) -> bool:
        frequency = self.cfg.rebalance_frequency
        if frequency == "daily":
            return True
        if frequency == "alternate":
            return (position - 1) % 2 == 0
        if position == 1:
            return True
        current, previous = pd.Timestamp(dates[position]), pd.Timestamp(dates[position - 1])
        if frequency == "weekly":
            return current.to_period("W") != previous.to_period("W")
        if frequency == "monthly":
            return current.to_period("M") != previous.to_period("M")
        if frequency == "quarterly":
            return current.to_period("Q") != previous.to_period("Q")
        raise ValueError(f"不支持的调仓频率：{frequency}")

    def run(self):
        dates = self.d["twap"].index
        for i, date in enumerate(dates):
            if self.cancel_check and self.cancel_check():
                raise BacktestCancelled("回测已由用户终止")
            if self.progress_callback and (i == 0 or i % 10 == 0 or i == len(dates) - 1):
                self.progress_callback(i + 1, len(dates), date)
            if i == 0:
                # 留出初始资产基线；第一笔交易必须等到已有上一交易日因子。
                self.daily.append([date, self.cash, 0.0, self.cash, 0])
                if self.daily_callback:
                    self.daily_callback(date, [], [], self.cash, self.cash)
                continue
            prev_date = dates[i - 1]
            self._corporate_actions(date, prev_date)
            prev_close = self._close_prices(prev_date)
            scores = self.d["factor"].loc[prev_date].replace([np.inf, -np.inf], np.nan)
            in_pool = self.d["pool"].loc[prev_date].fillna(False).astype(bool)
            ranked = scores[in_pool & scores.notna()].sort_values(ascending=False)
            if self.cfg.selection_mode == "top_quantile":
                desired_count = max(1, int(math.ceil(len(ranked) * self.cfg.rotation_quantile)))
            else:
                desired_count = self.cfg.holding_count
            desired_codes = ranked.index[:desired_count]
            previous_total = float(self.daily[-1][3])
            self.peak_asset = max(self.peak_asset, previous_total)
            current_drawdown = previous_total / self.peak_asset - 1 if self.peak_asset > 0 else 0.0
            signal_enabled = (bool(self.d["trade_enabled"].get(prev_date, True))
                              if "trade_enabled" in self.d else True)
            drawdown_enabled = (self.cfg.max_drawdown_limit is None or
                                current_drawdown > -self.cfg.max_drawdown_limit)
            # A calendar schedule is only a permission to trade. Sparse monthly
            # factors must never trigger a liquidation on a day without a new
            # signal merely because that day matches the selected schedule.
            signal_available = bool(scores.notna().any())
            scheduled_rebalance = self._scheduled_rebalance(i, dates) and signal_available
            trade_gate_enabled = signal_enabled and drawdown_enabled
            rebalance_enabled = trade_gate_enabled and scheduled_rebalance
            risk_off = (bool(self.d["risk_off"].get(prev_date, False))
                        if "risk_off" in self.d else False)
            qp_optimizer = self.d.get("drawdown_optimizer")
            if qp_optimizer is not None:
                totals = pd.Series([row[3] for row in self.daily], dtype=float)
                recent_returns = totals.pct_change(fill_method=None).dropna()
                previous_exposure = (self.daily[-1][2] / previous_total
                                     if self.positions and previous_total > 0
                                     else self.target_exposure)
                qp_decision = qp_optimizer.optimize_exposure(
                    current_drawdown, recent_returns, previous_exposure, risk_off)
                target_exposure = float(qp_decision["target_exposure"])
            else:
                target_exposure = 1.0
                qp_decision = {"qp_status": "disabled", "qp_observations": 0,
                               "trailing_mean": np.nan, "downside_second_moment": np.nan,
                               "risk_multiplier": 1.0, "unconstrained_exposure": 1.0}

            if not rebalance_enabled:
                target = []
            elif not self.positions:
                target = [c for c in desired_codes if self._tradeable(date, c, "buy", prev_close)]
            elif self.cfg.selection_mode == "top_quantile":
                # Diagnostic decile portfolio: fully rebuild the book so prior
                # weights and turnover caps cannot contaminate ranking quality.
                for code in list(self.positions):
                    self._sell(date, code, prev_close)
                target = [c for c in desired_codes if c not in self.positions and self._tradeable(
                    date, c, "buy", prev_close)]
            elif self.cfg.selection_mode == "decile_rotation":
                held_rank = scores.reindex(self.positions).fillna(-np.inf).sort_values()
                replacement_count = min(
                    self.cfg.max_replacements,
                    max(1, int(math.ceil(len(self.positions) * self.cfg.rotation_quantile))))
                sold = 0
                for code in held_rank.index[:replacement_count]:
                    sold += int(self._sell(date, code, prev_close))
                candidate_count = int(math.ceil(len(ranked) * self.cfg.rotation_quantile))
                missing = max(0, self.cfg.holding_count - len(self.positions))
                target = [c for c in ranked.index[:candidate_count]
                          if c not in self.positions and self._tradeable(date, c, "buy", prev_close)][:missing]
            else:
                held_rank = scores.reindex(self.positions).fillna(-np.inf).sort_values()
                retain = (self.d["retain"].loc[prev_date].reindex(self.positions).fillna(False).astype(bool)
                          if "retain" in self.d else pd.Series(False, index=list(self.positions)))
                sell_candidates = [c for c in held_rank.index
                                   if c not in ranked.index[:self.cfg.holding_count] and not retain.get(c, False)]
                sold = 0
                for code in sell_candidates:
                    if sold >= self.cfg.max_replacements: break
                    sold += int(self._sell(date, code, prev_close))
                # Refill every missing slot. Using only ``sold`` here lets
                # delisted/suspended names permanently shrink the portfolio.
                missing = max(0, self.cfg.holding_count - len(self.positions))
                target = [c for c in ranked.index if c not in self.positions and self._tradeable(
                    date, c, "buy", prev_close)][:missing]

            close_now = self._close_prices(date)
            valuation_prices = {}
            for code, position in self.positions.items():
                price = close_now.get(code, np.nan)
                if np.isfinite(price):
                    self.last_close[code] = price
                valuation_prices[code] = self.last_close.get(code, position.avg_cost)
            equity_before = self.cash + sum(
                position.shares * valuation_prices[code] for code, position in self.positions.items()
            )
            current_exposure = (sum(
                position.shares * valuation_prices.get(code, position.avg_cost)
                for code, position in self.positions.items()) / equity_before
                if self.positions and equity_before > 0 else 0.0)
            if (qp_optimizer is not None and self.positions and
                    abs(target_exposure - current_exposure) >= 0.02):
                self._rebalance_exposure(
                    date, target_exposure, equity_before, prev_close)
                # Recompute after exposure trades and their fees.
                equity_before = self.cash + sum(
                    position.shares * valuation_prices.get(code, position.avg_cost)
                    for code, position in self.positions.items())
            self.target_exposure = target_exposure
            allocation = self._allocation_weights(scores, desired_codes)
            if self.cfg.selection_mode == "top_quantile" and target:
                # Unsellable stale positions can lock part of the equity. Spread
                # the remaining deployable cash across every executable target;
                # otherwise rank-order iteration spends it on the first names and
                # silently drops the rest of the requested quantile portfolio.
                held_value = sum(
                    position.shares * valuation_prices.get(code, position.avg_cost)
                    for code, position in self.positions.items())
                deployable = min(self.cash, max(
                    0.0, equity_before * target_exposure - held_value))
                target_weight = sum(allocation.get(code, 0.0) for code in target)
                for code in target:
                    relative_weight = (allocation.get(code, 0.0) / target_weight
                                       if target_weight > 1e-12 else 1.0 / len(target))
                    self._buy(date, code, deployable * relative_weight, prev_close)
            else:
                for code in target:
                    budget = equity_before * target_exposure * allocation.get(
                        code, 1.0 / max(self.cfg.holding_count, 1))
                    self._buy(date, code, budget, prev_close)

            stock_value = 0.0
            today_holdings = []
            for code, p in self.positions.items():
                px = close_now.get(code, np.nan)
                if np.isfinite(px):
                    self.last_close[code] = px
                else:
                    px = self.last_close.get(code, p.avg_cost)
                value = p.shares * px
                stock_value += value
                self.holdings.append([date, code, p.shares, px, value, p.avg_cost])
                today_holdings.append({"code": code, "shares": p.shares, "close": px,
                                       "market_value": value, "avg_cost": p.avg_cost})
            total = self.cash + stock_value
            self.daily.append([date, self.cash, stock_value, total, len(self.positions)])
            actual_exposure = stock_value / total if total > 0 else 0.0
            self.risk_events.append([
                date, signal_enabled, current_drawdown, drawdown_enabled,
                scheduled_rebalance, trade_gate_enabled, risk_off, target_exposure, actual_exposure,
                qp_decision["unconstrained_exposure"],
                qp_decision["qp_status"], qp_decision["qp_observations"],
                qp_decision["trailing_mean"], qp_decision["downside_second_moment"],
                qp_decision["risk_multiplier"]])
            if self.daily_callback:
                today_orders = [
                    {"date": x[0].strftime("%Y-%m-%d"), "code": x[1], "side": x[2],
                     "shares": x[3], "price": x[4], "fee": x[5], "gross": x[6]}
                    for x in self.orders if x[0] == date
                ]
                self.daily_callback(date, today_orders, today_holdings, total, self.cfg.initial_cash)
        return BacktestResult(self, self.d["benchmark"])


class BacktestResult:
    def __init__(self, e: BacktestEngine, benchmark):
        self.account = pd.DataFrame(e.daily, columns=["date", "cash", "stock_value", "total_asset", "holding_count"]).set_index("date")
        self.orders = pd.DataFrame(e.orders, columns=["date", "code", "side", "shares", "price", "fee", "gross"])
        self.trades = pd.DataFrame(e.closed, columns=["code", "buy_date", "sell_date", "buy_price", "sell_price", "shares", "pnl", "holding_days"])
        self.holdings = pd.DataFrame(e.holdings, columns=["date", "code", "shares", "close", "market_value", "avg_cost"])
        self.risk_events = pd.DataFrame(e.risk_events, columns=[
            "date", "signal_enabled", "prior_drawdown", "drawdown_enabled",
            "scheduled_rebalance", "trade_enabled",
            "risk_off", "target_exposure", "actual_exposure", "unconstrained_exposure",
            "qp_status", "qp_observations", "trailing_mean", "downside_second_moment",
            "risk_multiplier"])
        self.benchmark = benchmark.reindex(self.account.index).ffill()
