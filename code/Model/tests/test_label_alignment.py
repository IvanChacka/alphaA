from imports import *
import pytest
from env import BACKTEST_DIR

if str(BACKTEST_DIR) not in sys.path:
    sys.path.insert(0, str(BACKTEST_DIR))
from rolling_ml.label_builder import LabelBuilder

def test_label_alignment():
    dates=pd.bdate_range("2024-01-02",periods=5);prices=pd.DataFrame({"000001":[10,11,12,13,14]},index=dates)
    labels=LabelBuilder().build(prices);row=labels[(labels.factor_date==dates[0])&(labels.symbol=="000001")].iloc[0]
    assert np.isclose(row.label,12/11-1);assert row.label_entry_date==dates[1];assert row.label_exit_date==dates[2]
    tail = labels[labels.factor_date.isin(dates[-1:])]
    assert tail.label.isna().all()


def test_label_alignment_follows_sparse_factor_frequency():
    dates = pd.bdate_range("2024-01-02", periods=6)
    prices = pd.DataFrame({"000001": [10, 11, 12, 13, 14, 15]}, index=dates)
    factor_dates = dates[[0, 2, 4]]
    labels = LabelBuilder().build(prices, factor_dates)
    row = labels[(labels.factor_date == dates[0]) & (labels.symbol == "000001")].iloc[0]
    assert row.label_entry_date == dates[1]
    assert row.label_exit_date == dates[3]
    assert np.isclose(row.label, 13 / 11 - 1)
    assert labels[labels.factor_date.isin(factor_dates[-1:])].label.isna().all()


def test_calendar_month_end_label_uses_next_tradable_days():
    dates = pd.to_datetime(["2024-05-30", "2024-06-03", "2024-06-28", "2024-07-01"])
    prices = pd.DataFrame({"000001": [10.0, 11.0, 12.0, 13.0]}, index=dates)
    factor_dates = pd.to_datetime(["2024-05-31", "2024-06-30"])

    labels = LabelBuilder().build(prices, factor_dates)
    row = labels.loc[labels.factor_date == pd.Timestamp("2024-05-31")].iloc[0]

    assert row.label_entry_date == pd.Timestamp("2024-06-03")
    assert row.label_exit_date == pd.Timestamp("2024-07-01")
    assert row.label == pytest.approx(13 / 11 - 1)


def test_price_dates_removes_forward_filled_holiday_rows():
    from data_loader import MarketData

    dates = pd.to_datetime(["2024-04-30", "2024-05-01", "2024-05-02",
                            "2024-05-03", "2024-05-06"])
    prices = pd.DataFrame({"000001": [10.0, 10.0, 10.5, 10.5, 11.0],
                           "000002": [20.0, 20.0, 20.5, 20.5, 21.0]}, index=dates)

    actual = MarketData._price_dates(prices)

    assert actual.tolist() == [pd.Timestamp("2024-04-30"), pd.Timestamp("2024-05-02"),
                               pd.Timestamp("2024-05-06")]
