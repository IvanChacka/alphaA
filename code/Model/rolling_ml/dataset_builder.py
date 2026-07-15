from imports import *
from rolling_ml.data_loader import ExistingDataAdapter


@dataclass
class DatasetBundle:
    data: pd.DataFrame
    features: list[str]
    daily_counts: pd.DataFrame
    missing_rates: pd.DataFrame


class DatasetBuilder:
    """合并因子和标签，同时保留测试期末尾尚无标签的预测样本。"""

    def build(self, factors: pd.DataFrame, labels: pd.DataFrame) -> DatasetBundle:
        data = factors.merge(labels, on=["factor_date", "symbol"], how="left", validate="one_to_one")
        features = ExistingDataAdapter.feature_columns(data)
        if not features:
            raise ValueError("没有可用的数值特征")
        counts = data.groupby("factor_date").size().rename("sample_count").reset_index()
        missing = data[features].isna().mean().rename_axis("feature").reset_index(name="missing_rate")
        return DatasetBundle(data.sort_values(["factor_date", "symbol"]), features, counts, missing)
