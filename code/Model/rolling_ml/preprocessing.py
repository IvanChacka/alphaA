from imports import *


class FeaturePreprocessor:
    """统一缺失处理后，按模型需要执行标准化。

    不执行MAD或缺失值填充；线性模型使用StandardScaler，XGBoost直通。
    非有限值直接报错，防止模型端静默修补上游数据。
    """

    def __init__(self, scale: bool = True):
        self.scale = scale
        self.pipeline = StandardScaler() if scale else None

    @staticmethod
    def _array(x: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = x.to_numpy(dtype=np.float32, copy=False) if isinstance(x, pd.DataFrame) else np.asarray(x, dtype=np.float32)
        if not np.isfinite(values).all():
            nan_count = int(np.isnan(values).sum())
            inf_count = int(np.isinf(values).sum())
            raise ValueError(f"输入因子包含非有限值：NaN={nan_count}, Inf={inf_count}；请在因子数据模块完成清洗")
        return values

    @staticmethod
    def cross_sectional(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        """直接返回上游已处理的因子，不再进行日截面二次加工。"""
        return frame.loc[:, features]

    def fit(self, x: pd.DataFrame | np.ndarray):
        values = self._array(x)
        if self.pipeline is not None:
            self.pipeline.fit(values)
        return self

    def transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = self._array(x)
        return self.pipeline.transform(values).astype(np.float32, copy=False) if self.pipeline is not None else values

    def fit_transform(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        values = self._array(x)
        return self.pipeline.fit_transform(values).astype(np.float32, copy=False) if self.pipeline is not None else values

    def save(self, path: Path | str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path | str):
        return joblib.load(path)
