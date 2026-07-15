from imports import *


class ScoreOptimizer(ABC):
    """在预测分数进入BackTest前施加横截面约束。"""

    name = "base"

    @abstractmethod
    def transform(self, scores: pd.DataFrame, pool: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """返回优化后的分数矩阵和逐日审计记录。"""


class NoOptimizer(ScoreOptimizer):
    name = "none"

    def transform(self, scores: pd.DataFrame, pool: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return scores.copy(), pd.DataFrame()
