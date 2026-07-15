from imports import *


@dataclass(frozen=True)
class PeriodSplit:
    name: str
    train_cutoff: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp


class TimeSplitter:
    def __init__(self, test_year: int = 2024):
        self.test_year = test_year

    def quarters(self, calendar: pd.DatetimeIndex) -> list[PeriodSplit]:
        dates = calendar[calendar.year == self.test_year]
        result: list[PeriodSplit] = []
        for quarter in range(1, 5):
            prediction = dates[dates.quarter == quarter]
            if prediction.empty:
                continue
            prior = calendar[calendar < prediction.min()]
            result.append(PeriodSplit(
                f"{self.test_year}Q{quarter}", prior.max(), prediction.min(), prediction.max()
            ))
        return result

    @staticmethod
    def training(data: pd.DataFrame, split: PeriodSplit, start: str,
                 mode: str = "expanding", years: int = 4) -> pd.DataFrame:
        lower = pd.Timestamp(start)
        if mode == "fixed":
            lower = split.train_cutoff - pd.DateOffset(years=years) + pd.Timedelta(days=1)
        mask = (
            (data.factor_date >= lower)
            & data.label.notna()
            & data.label_exit_date.notna()
            & (data.label_exit_date <= split.train_cutoff)
        )
        return data.loc[mask].copy()

    @staticmethod
    def prediction(data: pd.DataFrame, split: PeriodSplit) -> pd.DataFrame:
        mask = data.factor_date.between(split.prediction_start, split.prediction_end)
        return data.loc[mask].copy()

    def validation_folds(self, data: pd.DataFrame, year: int = 2023):
        folds = []
        annual = data[data.factor_date.dt.year == year]
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
            raise AssertionError(f"{split.name} 训练集或预测集为空")
        assert train.label_exit_date.max() <= split.train_cutoff
        assert train.factor_date.max() < test.factor_date.min()
        assert test.factor_date.min() >= split.prediction_start
        assert test.factor_date.max() <= split.prediction_end
