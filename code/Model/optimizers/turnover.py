from imports import *


class NoTurnoverOptimizer:
    name = "none"
    max_turnover_ratio: float | None = None

    def build_retain(self, predictions: pd.DataFrame, dates: pd.DatetimeIndex,
                     symbols: pd.Index) -> tuple[pd.DataFrame | None, pd.DataFrame]:
        return None, pd.DataFrame()


class TurnoverLimitOptimizer(NoTurnoverOptimizer):
    """Generic replacement cap that works with every model score table."""

    name = "turnover_limit"

    def __init__(self, max_turnover_ratio: float = .30):
        if not 0 < max_turnover_ratio <= 1:
            raise ValueError("最大换手比例必须在0到1之间")
        self.max_turnover_ratio = float(max_turnover_ratio)


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
        if "retain" in predictions:
            frame = predictions[["factor_date", "symbol", "retain"]].copy()
            frame["sell_votes"] = np.nan
        elif "sell_votes" in predictions:
            frame = predictions[["factor_date", "symbol", "sell_votes"]].copy()
            frame["retain"] = (
                frame.sell_votes.fillna(0).astype(float) < self.sell_confirmations)
        else:
            raise ValueError("惰性换手优化器需要模型输出retain或sell_votes")
        frame["factor_date"] = pd.to_datetime(frame.factor_date)
        frame["symbol"] = frame.symbol.astype(str).str.zfill(6)
        frame["retain"] = frame.retain.fillna(False).astype(bool)
        wide = frame.pivot_table(index="factor_date", columns="symbol", values="retain",
                                 aggfunc="last").reindex(index=dates, columns=symbols,
                                                         fill_value=False).astype(bool)
        audit = frame.groupby("factor_date", as_index=False).agg(
            stock_count=("symbol", "nunique"), retained_count=("retain", "sum"),
            average_sell_votes=("sell_votes", "mean"))
        audit["sell_confirmations"] = self.sell_confirmations
        audit["max_turnover_ratio"] = self.max_turnover_ratio
        return wide, audit
