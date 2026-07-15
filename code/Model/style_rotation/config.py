"""风格轮动策略可调参数；CLI 和网页端共用同一组校验规则。"""
from imports import *
from env import (STYLE_BUY_CONFIRMATIONS, STYLE_EXPLAINED_VARIANCE, STYLE_ICIR_WINDOW,
                 STYLE_MIN_COMPONENTS, STYLE_RESIDUAL_RIDGE_ALPHA, STYLE_RESIDUAL_WEIGHT,
                 STYLE_SELL_CONFIRMATIONS, STYLE_VOTE_QUANTILE)


@dataclass(frozen=True)
class StyleRuntimeConfig:
    """一次风格模型运行的参数快照。

    参数会写入 run_manifest，保证网页微调后的每次结果都可复现。
    """

    pca_variance: float = STYLE_EXPLAINED_VARIANCE
    pca_min_components: int = STYLE_MIN_COMPONENTS
    buy_confirmations: int = STYLE_BUY_CONFIRMATIONS
    sell_confirmations: int = STYLE_SELL_CONFIRMATIONS
    vote_quantile: float = STYLE_VOTE_QUANTILE
    residual_weight: float = STYLE_RESIDUAL_WEIGHT
    residual_alpha: float = STYLE_RESIDUAL_RIDGE_ALPHA
    icir_window: int = STYLE_ICIR_WINDOW

    def validate(self) -> "StyleRuntimeConfig":
        if not 0.50 <= self.pca_variance <= 0.999:
            raise ValueError("PCA累计解释率必须在0.50到0.999之间")
        if self.pca_min_components < 1:
            raise ValueError("PCA最少成分数必须大于0")
        if self.buy_confirmations < 1 or self.sell_confirmations < 1:
            raise ValueError("买入/卖出确认风格数必须大于0")
        if not 0.50 < self.vote_quantile < 1.0:
            raise ValueError("看多/看空分位必须在0.50到1之间")
        if not 0.0 <= self.residual_weight <= 10.0:
            raise ValueError("残差预测权重必须在0到10之间")
        if self.residual_alpha <= 0:
            raise ValueError("残差Ridge alpha必须大于0")
        if self.icir_window < 10:
            raise ValueError("滚动ICIR窗口不能少于10个交易日")
        return self
