from imports import *


class CrossSectionalPreprocessor:
    """Standardize each daily cross-section first, then fill missing values with 0.

    Means and variances use observed stocks only. Missing observations therefore
    do not affect cross-sectional statistics and become neutral only after the
    observed values have been standardized.
    """

    def transform(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        values = frame[features].to_numpy(dtype=np.float32, copy=False)
        inf_count = int(np.isinf(values).sum())
        if inf_count:
            raise ValueError(f"输入因子包含无穷值：Inf={inf_count}；请先修复上游因子文件")

        parts = []
        for _, group in frame.groupby("factor_date", sort=True):
            daily = group[features].to_numpy(dtype=np.float32, copy=True)
            observed = ~np.isnan(daily)
            counts = observed.sum(axis=0)
            safe_counts = np.maximum(counts, 1)
            observed_values = np.where(observed, daily, 0.0)
            mean = observed_values.sum(axis=0) / safe_counts
            centered = np.where(observed, daily - mean, 0.0)
            variance = np.square(centered).sum(axis=0) / safe_counts
            std = np.sqrt(variance)
            denominator = np.where(std > 1e-12, std, 1.0)
            standardized = np.where(observed, centered / denominator, np.nan)
            standardized = np.where(np.isnan(standardized), 0.0, standardized)

            out = group[["factor_date", "symbol"]].copy()
            out[features] = standardized.astype(np.float32, copy=False)
            parts.append(out)
        columns = ["factor_date", "symbol", *features]
        return pd.concat(parts, ignore_index=True) if parts else frame[columns].copy()
