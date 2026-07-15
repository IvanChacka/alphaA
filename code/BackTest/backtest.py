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
        self.positions: dict[str, Position] = {}
        # 停牌或退市期间用最后一个有效收盘价估值；只影响账面估值，不允许交易。
        self.last_close: dict[str, float] = {}
        self.orders, self.closed, self.daily, self.holdings = [], [], [], []

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
        if np.isfinite(yesterday) and yesterday > 0:
            limit = .20 if code.startswith(("300", "688")) else .10
            change = twap / yesterday - 1
            if change >= limit - 1e-6 or change <= -limit + 1e-6:
                return False
        return True

    def _sell(self, date, code, prev_close):
        if not self._tradeable(date, code, "sell", prev_close): return False
        p = self.positions[code]
        price = self.actual(self.d["twap"].at[date, code], self.d["adj"].at[date, code])
        gross, fee = p.shares * price, p.shares * price * self.cfg.fee_rate
        self.cash += gross - fee
        pnl = (price - p.avg_cost) * p.shares - fee
        self.orders.append([date, code, "SELL", p.shares, price, fee, gross])
        self.closed.append([code, p.opened, date, p.avg_cost, price, p.shares, pnl,
                            (date - p.opened).days])
        del self.positions[code]
        self.last_close.pop(code, None)
        return True

    def _buy(self, date, code, budget, prev_close):
        if not self._tradeable(date, code, "buy", prev_close): return False
        price = self.actual(self.d["twap"].at[date, code], self.d["adj"].at[date, code])
        lot = self.lot(code)
        shares = math.floor(min(budget, self.cash) / (price * (1 + self.cfg.fee_rate)) / lot) * lot
        if shares <= 0: return False
        gross, fee = shares * price, shares * price * self.cfg.fee_rate
        self.cash -= gross + fee
        self.positions[code] = Position(shares, price + fee / shares, date)
        self.orders.append([date, code, "BUY", shares, price, fee, gross])
        return True

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

            if not self.positions:
                target = [c for c in ranked.index if self._tradeable(date, c, "buy", prev_close)][:self.cfg.holding_count]
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
                target = [c for c in ranked.index if c not in self.positions and self._tradeable(date, c, "buy", prev_close)][:sold]

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
            slots = max(self.cfg.holding_count, 1)
            for code in target:
                self._buy(date, code, equity_before / slots, prev_close)

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
        self.benchmark = benchmark.reindex(self.account.index).ffill()
