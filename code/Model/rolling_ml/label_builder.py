from imports import *


class LabelBuilder:
    """Map a t0 signal to its actually tradable next-period return."""

    def build(self, real_twap: pd.DataFrame, factor_dates=None) -> pd.DataFrame:
        if not isinstance(real_twap.index, pd.DatetimeIndex) or real_twap.index.has_duplicates:
            raise ValueError("TWAP索引必须是唯一的交易日DatetimeIndex")
        prices = real_twap.sort_index()
        if not prices.index.is_monotonic_increasing:
            raise ValueError("TWAP交易日必须升序")
        if factor_dates is None:
            signal_dates = prices.index
        else:
            signal_dates = pd.DatetimeIndex(pd.to_datetime(factor_dates)).unique().sort_values()
        entry_dates = pd.Series(pd.NaT, index=signal_dates, dtype="datetime64[ns]")
        exit_dates = pd.Series(pd.NaT, index=signal_dates, dtype="datetime64[ns]")
        if len(signal_dates) > 1:
            active = signal_dates[:-1]
            entry_pos = prices.index.searchsorted(active, side="right")
            exit_pos = prices.index.searchsorted(signal_dates[1:], side="right")
            valid = (entry_pos < len(prices.index)) & (exit_pos < len(prices.index))
            entry_dates.loc[active[valid]] = prices.index[entry_pos[valid]]
            exit_dates.loc[active[valid]] = prices.index[exit_pos[valid]]
        valid_dates = exit_dates.dropna().index
        labels = prices.reindex(exit_dates.loc[valid_dates].values).set_axis(valid_dates).div(
            prices.reindex(entry_dates.loc[valid_dates].values).set_axis(valid_dates)
        ).sub(1)
        labels = labels.reindex(signal_dates)
        entry = entry_dates.reindex(signal_dates)
        exit_date = exit_dates.reindex(signal_dates)
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
        if (sample.entry_offset >= sample.exit_offset).any():
            raise AssertionError("标签进入日期必须早于退出日期")
        tail_dates = pd.DatetimeIndex(labels.factor_date.unique()).sort_values()[-1:]
        tail = labels[labels.factor_date.isin(tail_dates)]
        if tail.label.notna().any():
            raise AssertionError("标签尾部两个交易日应为空")
        return sample
