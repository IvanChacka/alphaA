from imports import *


class CrossSectionalPreprocessor:
    """逐日截面标准化；不做MAD、不填充、不跨日期使用统计量。"""

    def transform(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        values = frame[features].to_numpy(dtype=np.float32, copy=False)
        if not np.isfinite(values).all():
            nan_count = int(np.isnan(values).sum())
            inf_count = int(np.isinf(values).sum())
            raise ValueError(f"输入因子包含非有限值：NaN={nan_count}, Inf={inf_count}；请先修复上游因子文件")
        parts = []
        for _, group in frame.groupby("factor_date", sort=True):
            daily = group[features].to_numpy(dtype=np.float32, copy=True)
            mean = daily.mean(axis=0)
            std = daily.std(axis=0)
            daily = (daily - mean) / np.where(std > 1e-12, std, 1.0)
            out = group[["factor_date", "symbol"]].copy()
            out[features] = daily.astype(np.float32, copy=False)
            parts.append(out)
        return pd.concat(parts, ignore_index=True) if parts else frame[["factor_date", "symbol", *features]].copy()
