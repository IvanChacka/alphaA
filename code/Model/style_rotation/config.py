"""风格轮动策略可调参数；CLI 和网页端共用同一组校验规则。"""
from imports import *
from env import (STYLE_BUY_CONFIRMATIONS, STYLE_EXPLAINED_VARIANCE, STYLE_ICIR_WINDOW,
                 STYLE_MIN_COMPONENTS, STYLE_RESIDUAL_RIDGE_ALPHA, STYLE_RESIDUAL_WEIGHT,
                 STYLE_VOTE_QUANTILE, STYLE_XGB_MAX_DEPTH,
                 STYLE_PCA_ALIGNMENT_THRESHOLD, STYLE_CONFIDENCE_RANKIC_WINDOW,
                 STYLE_CONFIDENCE_RANKIC_THRESHOLD, STYLE_ENTRY_RANK, STYLE_EXIT_RANK,
                 STYLE_PCA_EXPLAINED_THRESHOLD, STYLE_HOLDING_COUNT,
                 STYLE_ROTATION_QUANTILE, STYLE_REBALANCE_FREQUENCY)


@dataclass(frozen=True)
class StyleRuntimeConfig:
    """一次风格模型运行的参数快照。

    参数会写入 run_manifest，保证网页微调后的每次结果都可复现。
    """

    pca_variance: float = STYLE_EXPLAINED_VARIANCE
    pca_min_components: int = STYLE_MIN_COMPONENTS
    buy_confirmations: int = STYLE_BUY_CONFIRMATIONS
    vote_quantile: float = STYLE_VOTE_QUANTILE
    residual_weight: float = STYLE_RESIDUAL_WEIGHT
    residual_alpha: float = STYLE_RESIDUAL_RIDGE_ALPHA
    icir_window: int = STYLE_ICIR_WINDOW
    xgb_max_depth: int = STYLE_XGB_MAX_DEPTH
    pca_alignment_threshold: float = STYLE_PCA_ALIGNMENT_THRESHOLD
    pca_explained_threshold: float = STYLE_PCA_EXPLAINED_THRESHOLD
    confidence_rankic_window: int = STYLE_CONFIDENCE_RANKIC_WINDOW
    confidence_rankic_threshold: float = STYLE_CONFIDENCE_RANKIC_THRESHOLD
    entry_rank: int = STYLE_ENTRY_RANK
    exit_rank: int = STYLE_EXIT_RANK
    holding_count: int = STYLE_HOLDING_COUNT
    rotation_quantile: float = STYLE_ROTATION_QUANTILE
    rebalance_frequency: str = STYLE_REBALANCE_FREQUENCY
    pca_fit_mode: str = "rolling"

    def validate(self) -> "StyleRuntimeConfig":
        if not 0.50 <= self.pca_variance <= 0.999:
            raise ValueError("PCA累计解释率必须在0.50到0.999之间")
        if self.pca_min_components < 1:
            raise ValueError("PCA最少成分数必须大于0")
        if self.buy_confirmations < 1:
            raise ValueError("买入确认风格数必须大于0")
        if not 0.50 < self.vote_quantile < 1.0:
            raise ValueError("看多/看空分位必须在0.50到1之间")
        if not 0.0 <= self.residual_weight <= 10.0:
            raise ValueError("残差预测权重必须在0到10之间")
        if self.residual_alpha <= 0:
            raise ValueError("残差Ridge alpha必须大于0")
        if self.icir_window < 10:
            raise ValueError("滚动ICIR窗口不能少于10个交易日")
        if not 1 <= self.xgb_max_depth <= 10:
            raise ValueError("风格XGBoost最大深度必须在1到10之间")
        if not 0.0 <= self.pca_alignment_threshold <= 1.0:
            raise ValueError("PCA载荷平均相似度门槛必须在0到1之间")
        if not 0.0 <= self.pca_explained_threshold <= 1.0:
            raise ValueError("PCA交易解释率门槛必须在0到1之间")
        if not 10 <= self.confidence_rankic_window <= 1000:
            raise ValueError("已实现RankIC窗口必须在10到1000个交易日之间")
        if not -1.0 <= self.confidence_rankic_threshold <= 1.0:
            raise ValueError("RankIC门槛必须在-1到1之间")
        if self.entry_rank < 1 or self.exit_rank < self.entry_rank:
            raise ValueError("退出排名必须不小于进入排名，且二者必须大于0")
        if self.holding_count < 1:
            raise ValueError("目标持仓数必须大于0")
        if not 0.0 < self.rotation_quantile <= 1.0:
            raise ValueError("调仓分位比例必须在0到1之间")
        if self.rebalance_frequency not in {"daily", "alternate", "weekly", "monthly", "quarterly"}:
            raise ValueError("调仓频率必须是日频、隔日、周频或月频")
        if self.pca_fit_mode not in {"rolling", "fixed"}:
            raise ValueError("PCA拟合方式必须是滚动训练或固定训练期")
        return self
