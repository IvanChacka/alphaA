from imports import *


class FeaturePreprocessor:
    """Prepare factors without letting missing observations affect scaling.

    Linear models fit ``StandardScaler`` on the training sample while ignoring
    NaNs, then replace remaining missing values with the neutral standardized
    value 0. Models without a second scaling pass keep upstream standardized
    observations unchanged and fill NaNs only at the model-input boundary.
    """

    def __init__(self, scale: bool = True):
        self.scale = scale
        self.pipeline = StandardScaler() if scale else None

    @staticmethod
    def _array(x: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = (
            x.to_numpy(dtype=np.float32, copy=False)
            if isinstance(x, pd.DataFrame)
            else np.asarray(x, dtype=np.float32)
        )
        inf_count = int(np.isinf(values).sum())
        if inf_count:
            raise ValueError(f"输入因子包含无穷值：Inf={inf_count}；请先修复上游因子文件")
        return values

    @staticmethod
    def _fill_standardized_missing(values: np.ndarray) -> np.ndarray:
        """Fill only NaNs; infinities must never be repaired silently."""
        if np.isinf(values).any():
            raise ValueError("标准化结果包含无穷值，请检查训练样本的因子分布")
        return np.where(np.isnan(values), 0.0, values).astype(np.float32, copy=False)

    @staticmethod
    def cross_sectional(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """Return upstream factors unchanged; partial NaNs are intentionally kept."""
        return frame.loc[:, features]

    def fit(self, x: pd.DataFrame | np.ndarray):
        values = self._array(x)
        if self.pipeline is not None:
            all_missing = np.isnan(values).all(axis=0)
            if all_missing.any():
                columns = (
                    list(np.asarray(x.columns)[all_missing])
                    if isinstance(x, pd.DataFrame)
                    else np.flatnonzero(all_missing).tolist()
                )
                raise ValueError(f"训练集中存在全缺失因子列，无法标准化：{columns}")
            self.pipeline.fit(values)
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = self._array(x)
        transformed = self.pipeline.transform(values) if self.pipeline is not None else values
        return self._fill_standardized_missing(transformed)

    def fit_transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str):
        return joblib.load(path)
