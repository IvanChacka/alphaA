from imports import *
from env import INDUSTRY_DATA_CANDIDATES, OPTIMIZER_NAMES
from optimizers.base import NoOptimizer, ScoreOptimizer
from optimizers.industry_neutral import IndustryNeutralOptimizer


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
