from imports import *


class NoTurnoverOptimizer:
    name = "none"
    max_turnover_ratio: float | None = None

    def build_retain(self, predictions: pd.DataFrame, dates: pd.DatetimeIndex,
                     symbols: pd.Index) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        return None, pd.DataFrame()


class LazyTurnoverOptimizer:
    """把风格投票转换为持仓保留信号，并限制每日最大换手比例。"""

    name = "lazy_turnover"

    def __init__(self, sell_confirmations: int = 2, max_turnover_ratio: float = .30):
        if sell_confirmations < 1:
            raise ValueError("卖出确认风格数必须大于0")
        if not 0 < max_turnover_ratio <= 1:
            raise ValueError("最大换手比例必须在0到1之间")
        self.sell_confirmations = int(sell_confirmations)
        self.max_turnover_ratio = float(max_turnover_ratio)

    def build_retain(self, predictions: pd.DataFrame, dates: pd.DatetimeIndex,
                     symbols: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
        if "sell_votes" not in predictions:
            raise ValueError("惰性换手优化器需要模型输出sell_votes；当前模型不支持该优化器")
        frame = predictions[["factor_date", "symbol", "sell_votes"]].copy()
        frame["factor_date"] = pd.to_datetime(frame.factor_date)
        frame["symbol"] = frame.symbol.astype(str).str.zfill(6)
        frame["retain"] = frame.sell_votes.fillna(0).astype(float) < self.sell_confirmations
        wide = frame.pivot_table(index="factor_date", columns="symbol", values="retain",
                                 aggfunc="last").reindex(index=dates, columns=symbols,
                                                         fill_value=False).astype(bool)
        audit = frame.groupby("factor_date", as_index=False).agg(
            stock_count=("symbol", "nunique"), retained_count=("retain", "sum"),
            average_sell_votes=("sell_votes", "mean"))
        audit["sell_confirmations"] = self.sell_confirmations
        audit["max_turnover_ratio"] = self.max_turnover_ratio
        return wide, audit
