from imports import *
from env import INDUSTRY_DATA_CANDIDATES, OPTIMIZER_NAMES
from optimizers.base import NoOptimizer, ScoreOptimizer
from optimizers.industry_neutral import IndustryNeutralOptimizer
from optimizers.turnover import LazyTurnoverOptimizer, NoTurnoverOptimizer
from optimizers.drawdown import NoDrawdownOptimizer, QuadraticDrawdownOptimizer


def resolve_industry_path() -> Path:
    path = next((Path(candidate) for candidate in INDUSTRY_DATA_CANDIDATES if Path(candidate).exists()), None)
    if path is None:
        names = " 或 ".join(str(path) for path in INDUSTRY_DATA_CANDIDATES)
        raise FileNotFoundError(f"行业中性优化需要行业分类文件：{names}")
    return path


def create_optimizer(name: str) -> ScoreOptimizer:
    if name not in OPTIMIZER_NAMES:
        raise ValueError(f"未知优化器：{name}")
    if name == "none":
        return NoOptimizer()
    return IndustryNeutralOptimizer(resolve_industry_path())


def create_turnover_optimizer(name: str, sell_confirmations: int = 2,
                              max_turnover_ratio: float = .30):
    if name == "none":
        return NoTurnoverOptimizer()
    if name == "lazy_turnover":
        return LazyTurnoverOptimizer(sell_confirmations, max_turnover_ratio)
    raise ValueError(f"未知换手率优化器：{name}")


def create_drawdown_optimizer(name: str, max_drawdown_limit: float = .20):
    if name == "none":
        return NoDrawdownOptimizer()
    if name in {"quadratic_drawdown", "max_drawdown"}:
        return QuadraticDrawdownOptimizer(max_drawdown_limit)
    raise ValueError(f"未知最大回撤优化器：{name}")
