from imports import *


class LabelBuilder:
    """在因子日t0保存 real_twap(t2) / real_twap(t1) - 1。"""

    def build(self, real_twap: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(real_twap.index, pd.DatetimeIndex) or real_twap.index.has_duplicates:
            raise ValueError("TWAP索引必须是唯一的交易日DatetimeIndex")
        prices = real_twap.sort_index()
        if not prices.index.is_monotonic_increasing:
            raise ValueError("TWAP交易日必须升序")
        labels = prices.shift(-2).div(prices.shift(-1)).sub(1)
        entry = pd.Series(prices.index, index=prices.index).shift(-1)
        exit_date = pd.Series(prices.index, index=prices.index).shift(-2)
        long = labels.stack(future_stack=True).rename("label").reset_index()
        long.columns = ["factor_date", "symbol", "label"]
        long["symbol"] = long.symbol.astype(str).str.zfill(6)
        long["label_entry_date"] = long.factor_date.map(entry)
        long["label_exit_date"] = long.factor_date.map(exit_date)
        return long

    @staticmethod
    def audit(labels: pd.DataFrame, real_twap: pd.DataFrame,
              sample_size: int = 100, seed: int = 42) -> pd.DataFrame:
        valid = labels.dropna(subset=["label", "label_entry_date", "label_exit_date"])
        sample = valid.sample(min(sample_size, len(valid)), random_state=seed).copy()
        sample["manual_label"] = [
            real_twap.at[row.label_exit_date, row.symbol] / real_twap.at[row.label_entry_date, row.symbol] - 1
            for row in sample.itertuples()
        ]
        sample["difference"] = sample.label - sample.manual_label
        calendar = real_twap.index
        positions = pd.Series(np.arange(len(calendar)), index=calendar)
        sample["entry_offset"] = sample.label_entry_date.map(positions) - sample.factor_date.map(positions)
        sample["exit_offset"] = sample.label_exit_date.map(positions) - sample.factor_date.map(positions)
        if not np.allclose(sample.label, sample.manual_label, equal_nan=True):
            raise AssertionError("标签价格计算审计失败")
        if not ((sample.entry_offset == 1) & (sample.exit_offset == 2)).all():
            raise AssertionError("标签没有按照交易日t1/t2对齐")
        tail = labels[labels.factor_date.isin(calendar[-2:])]
        if tail.label.notna().any():
            raise AssertionError("标签尾部两个交易日应为空")
        return sample
