from imports import *


@dataclass(frozen=True)
class PeriodSplit:
    name: str
    train_cutoff: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp


class TimeSplitter:
    def __init__(self, test_year: int = 2024, test_start: str | None = None,
                 test_end: str | None = None):
        self.test_year = test_year
        self.test_start = pd.Timestamp(test_start) if test_start else pd.Timestamp(f"{test_year}-01-01")
        self.test_end = pd.Timestamp(test_end) if test_end else pd.Timestamp(f"{test_year}-12-31")

    def quarters(self, calendar: pd.DatetimeIndex) -> list[PeriodSplit]:
        dates = calendar[(calendar >= self.test_start) & (calendar <= self.test_end)]
        result: list[PeriodSplit] = []
        for period in pd.period_range(self.test_start, self.test_end, freq="Q"):
            prediction = dates[dates.to_period("Q") == period]
            if prediction.empty:
                continue
            prior = calendar[calendar < prediction.min()]
            result.append(PeriodSplit(
                str(period), prior.max(), prediction.min(), prediction.max()
            ))
        return result

    def rolling_periods(self, calendar: pd.DatetimeIndex, frequency: str) -> list[PeriodSplit]:
        """Create model refit periods at the signal/rebalance frequency."""
        dates = calendar[(calendar >= self.test_start) & (calendar <= self.test_end)]
        if dates.empty:
            return []
        if frequency == "monthly":
            keys = dates.to_period("M")
        elif frequency == "quarterly":
            keys = dates.to_period("Q")
        elif frequency == "weekly":
            keys = dates.to_period("W")
        elif frequency == "alternate":
            keys = np.arange(len(dates)) // 2
        else:
            keys = dates.normalize()
        result = []
        for key in pd.unique(keys):
            prediction = dates[keys == key]
            prior = calendar[calendar < prediction.min()]
            if prior.empty:
                continue
            name = (str(key) if frequency in {"monthly", "quarterly"}
                    else prediction.min().strftime("%Y-%m-%d"))
            result.append(PeriodSplit(name, prior.max(), prediction.min(), prediction.max()))
        return result

    @staticmethod
    def training(data: pd.DataFrame, split: PeriodSplit, start: str,
                 mode: str = "expanding", years: int = 4,
                 end: str | None = None) -> pd.DataFrame:
        lower = pd.Timestamp(start)
        if mode == "fixed":
            lower = split.train_cutoff - pd.DateOffset(years=years) + pd.Timedelta(days=1)
        mask = (
            (data.factor_date >= lower)
            & (data.factor_date <= pd.Timestamp(end) if end else True)
            & data.label.notna()
            & data.label_exit_date.notna()
            & (data.label_exit_date <= split.train_cutoff)
        )
        return data.loc[mask].copy()

    @staticmethod
    def prediction(data: pd.DataFrame, split: PeriodSplit) -> pd.DataFrame:
        mask = data.factor_date.between(split.prediction_start, split.prediction_end)
        return data.loc[mask].copy()

    def validation_folds(self, data: pd.DataFrame, year: int = 2023,
                         end: str | None = None):
        folds = []
        annual = data[data.factor_date.dt.year == year]
        if end:
            annual = annual[annual.factor_date <= pd.Timestamp(end)]
        for quarter in range(1, 5):
            valid = annual[(annual.factor_date.dt.quarter == quarter) & annual.label.notna()]
            if valid.empty:
                continue
            validation_start = valid.factor_date.min()
            train = data[data.label.notna() & (data.label_exit_date < validation_start)]
            folds.append((f"{year}Q{quarter}", train.index, valid.index))
        return folds

    @staticmethod
    def assert_no_leakage(train: pd.DataFrame, test: pd.DataFrame, split: PeriodSplit) -> None:
        if train.empty or test.empty:
            raise AssertionError(f"{split.name} 训练集或预测集为空：训练={len(train)}，预测={len(test)}")
        assert train.label_exit_date.max() <= split.train_cutoff
        assert train.factor_date.max() < test.factor_date.min()
        assert test.factor_date.min() >= split.prediction_start
        assert test.factor_date.max() <= split.prediction_end
