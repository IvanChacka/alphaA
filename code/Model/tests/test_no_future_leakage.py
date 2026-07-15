from imports import *
from rolling_ml.time_splitter import PeriodSplit,TimeSplitter

def test_no_future_leakage():
    dates=pd.bdate_range("2023-12-20",periods=15);data=pd.DataFrame({"factor_date":dates[:-2],"label_exit_date":dates[2:],"label":1})
    split=PeriodSplit("2024Q1",pd.Timestamp("2023-12-29"),pd.Timestamp("2024-01-01"),pd.Timestamp("2024-03-31"))
    train=TimeSplitter.training(data,split,"2020-01-01");assert train.label_exit_date.max()<=split.train_cutoff


def test_validation_fold_uses_realized_label_boundary():
    dates = pd.bdate_range("2022-01-03", "2023-12-29")
    data = pd.DataFrame({"factor_date": dates[:-2], "label_entry_date": dates[1:-1],
                         "label_exit_date": dates[2:], "label": 1.0})
    folds = TimeSplitter(2024).validation_folds(data, 2023)
    assert len(folds) == 4
    for _, train_idx, valid_idx in folds:
        train, valid = data.loc[train_idx], data.loc[valid_idx]
        assert train.label_exit_date.max() < valid.factor_date.min()
