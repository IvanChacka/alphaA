"""预测未落入固定PCA风格子空间的原始因子残差。"""
from imports import *


class ResidualRidge:
    """低内存、可审计的滚动Ridge。

    PCA载荷张成已解释风格子空间。先用正交投影取其补空间，再在补空间上
    拟合股票t1到t2收益。训练矩阵以分块方式累计，不会复制整段百万行样本。
    """

    def __init__(self, loadings: np.ndarray, alpha: float = 0.05,
                 chunk_size: int = 200_000):
        if alpha <= 0:
            raise ValueError("alpha必须大于0")
        matrix = np.asarray(loadings, dtype=np.float64)
        if matrix.ndim != 2 or not matrix.size:
            raise ValueError("PCA载荷必须是非空二维矩阵")
        basis, _ = np.linalg.qr(matrix, mode="reduced")
        self.residual_projection = np.eye(matrix.shape[0]) - basis @ basis.T
        self.alpha = float(alpha)
        self.chunk_size = int(chunk_size)
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.train_rows_: int = 0
        self.max_label_exit_date_: pd.Timestamp | None = None

    def _residual(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype=np.float64) @ self.residual_projection

    def fit(self, frame: pd.DataFrame, features: list[str], label: str = "label",
            sample_weight: str | None = None) -> "ResidualRidge":
        dimension = len(features)
        gram = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
        rhs = np.zeros(dimension + 1, dtype=np.float64)
        weight_sum = 0.0
        valid_rows = 0
        for start in range(0, len(frame), self.chunk_size):
            part = frame.iloc[start:start + self.chunk_size]
            values = part[features].to_numpy(dtype=np.float64, copy=False)
            target = part[label].to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(target) & np.isfinite(values).all(axis=1)
            if not valid.any():
                continue
            x = self._residual(values[valid])
            y = target[valid]
            w = (part[sample_weight].to_numpy(dtype=np.float64, copy=False)[valid]
                 if sample_weight else np.ones(len(y), dtype=np.float64))
            design = np.column_stack([x, np.ones(len(x), dtype=np.float64)])
            gram += design.T @ (design * w[:, None])
            rhs += design.T @ (y * w)
            weight_sum += float(w.sum())
            valid_rows += len(y)
        if valid_rows == 0 or weight_sum <= 0:
            raise ValueError("残差Ridge没有可用训练样本")
        gram /= weight_sum
        rhs /= weight_sum
        penalty = np.eye(dimension + 1) * self.alpha
        penalty[-1, -1] = 0.0
        solution = np.linalg.solve(gram + penalty, rhs)
        self.coef_, self.intercept_ = solution[:-1], float(solution[-1])
        self.train_rows_ = valid_rows
        if "label_exit_date" in frame:
            self.max_label_exit_date_ = pd.to_datetime(frame.loc[frame[label].notna(), "label_exit_date"]).max()
        return self

    def predict(self, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("残差Ridge尚未拟合")
        result = np.empty(len(frame), dtype=np.float64)
        for start in range(0, len(frame), self.chunk_size):
            part = frame.iloc[start:start + self.chunk_size]
            result[start:start + len(part)] = (
                self._residual(part[features].to_numpy(dtype=np.float64, copy=False)) @ self.coef_
                + self.intercept_)
        return result

    def coefficient_frame(self, features: list[str]) -> pd.DataFrame:
        if self.coef_ is None:
            raise RuntimeError("残差Ridge尚未拟合")
        return pd.DataFrame({"factor": [*features, "intercept"],
                             "coefficient": [*self.coef_, self.intercept_]})
