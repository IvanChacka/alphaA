from imports import *
from env import STYLE_EXPLAINED_VARIANCE, STYLE_MIN_COMPONENTS


def varimax(loadings: np.ndarray, gamma: float = 1.0, max_iter: int = 200,
            tolerance: float = 1e-7) -> tuple[np.ndarray, np.ndarray, bool]:
    """正交Varimax旋转。"""
    rows, cols = loadings.shape
    rotation = np.eye(cols)
    previous = 0.0
    for _ in range(max_iter):
        product = loadings @ rotation
        u, singular, vh = np.linalg.svd(loadings.T @ (product ** 3 -
            (gamma / rows) * product @ np.diag(np.diag(product.T @ product))))
        rotation = u @ vh
        objective = singular.sum()
        if previous and objective - previous < tolerance * previous:
            return loadings @ rotation, rotation, True
        previous = objective
    return loadings @ rotation, rotation, False


class FixedPCAStyle:
    """以历史日截面相关矩阵的时间均值拟合一次，并永久冻结旋转载荷。"""

    def __init__(self, explained_variance: float = STYLE_EXPLAINED_VARIANCE,
                 min_components: int = STYLE_MIN_COMPONENTS):
        if not 0 < explained_variance <= 1:
            raise ValueError("PCA累计解释率必须在0到1之间")
        if min_components < 1:
            raise ValueError("PCA最少成分数必须大于0")
        self.explained_variance_target = float(explained_variance)
        self.min_components = int(min_components)
        self.features: list[str] = []
        self.loadings: np.ndarray | None = None
        self.explained_variance: np.ndarray | None = None
        self.rotation: np.ndarray | None = None

    def fit(self, frame: pd.DataFrame, features: list[str]):
        correlations = []
        for _, group in frame.groupby("factor_date", sort=True):
            x = group[features].to_numpy(dtype=np.float64)
            if len(x) > len(features):
                # 输入已逐日截面标准化，X.T@X/n即相关矩阵。某因子当日全缺失
                # 并填0时为零方差列，其相关项自然为0，避免np.corrcoef产生NaN
                # 后错误丢弃整个交易日。
                corr = (x.T @ x) / len(x)
                if np.isfinite(corr).all():
                    correlations.append(corr)
        if not correlations:
            raise ValueError("PCA拟合期没有有效的日截面相关矩阵")
        matrix = np.mean(correlations, axis=0)
        values, vectors = np.linalg.eigh(matrix)
        order = np.argsort(values)[::-1]
        values, vectors = np.maximum(values[order], 0), vectors[:, order]
        ratio = values / values.sum()
        k = int(np.searchsorted(np.cumsum(ratio), self.explained_variance_target) + 1)
        # 不再设置成分数上限：唯一上限是原始因子数。
        k = min(max(self.min_components, k), len(features))
        rotated, rotation, _ = varimax(vectors[:, :k] * np.sqrt(values[:k]))
        # 最大绝对载荷为正，消除特征向量符号不确定性。
        for col in range(k):
            anchor = int(np.argmax(np.abs(rotated[:, col])))
            if rotated[anchor, col] < 0: rotated[:, col] *= -1
        self.features, self.loadings = list(features), rotated
        self.explained_variance, self.rotation = ratio[:k], rotation
        return self

    @property
    def cumulative_explained_variance(self) -> float:
        return float(self.explained_variance.sum()) if self.explained_variance is not None else 0.0

    @property
    def style_names(self) -> list[str]:
        return [f"style_{i + 1}" for i in range(self.loadings.shape[1])]  # type: ignore[union-attr]

    def loading_frame(self) -> pd.DataFrame:
        if self.loadings is None: raise RuntimeError("PCA尚未拟合")
        return pd.DataFrame(self.loadings, index=self.features, columns=self.style_names).reset_index(names="factor")

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.loadings is None: raise RuntimeError("PCA尚未拟合")
        out = frame[["factor_date", "symbol"]].copy()
        scores = frame[self.features].to_numpy(dtype=np.float32) @ self.loadings.astype(np.float32)
        out[self.style_names] = scores
        out[self.style_names] = out.groupby("factor_date")[self.style_names].transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 1e-12 else 1.0))
        return out
