from imports import *
from rolling_ml.time_splitter import TimeSplitter
from rolling_ml.rolling_predict import RollingPredictor

def test_four_quarters_cover_year_without_overlap():
    cal=pd.bdate_range("2023-01-01","2024-12-31");parts=TimeSplitter(2024).quarters(cal)
    assert len(parts)==4
    dates=[]
    for p in parts:dates.extend(cal[(cal>=p.prediction_start)&(cal<=p.prediction_end)])
    assert len(dates)==len(set(dates))==sum(cal.year==2024)


def test_validation_year_can_follow_training_end_instead_of_test_year():
    dates = pd.to_datetime(["2022-01-03", "2022-04-01", "2022-07-01", "2022-10-10"])
    data = pd.DataFrame({"factor_date": dates, "label": [0.1, 0.2, 0.3, 0.4],
                         "label_exit_date": dates + pd.offsets.BDay(2)})
    training_end = "2022-12-31"
    validation_year = pd.Timestamp(training_end).year

    folds = TimeSplitter(2024).validation_folds(data, validation_year, training_end)

    assert [name for name, _, _ in folds] == ["2022Q1", "2022Q2", "2022Q3", "2022Q4"]


def test_prediction_quarters_support_cross_year_and_partial_final_year():
    calendar = pd.bdate_range("2024-01-01", "2026-09-11")
    parts = TimeSplitter(2024, "2024-01-01", "2026-09-11").quarters(calendar)

    assert parts[0].name == "2024Q1"
    assert parts[-1].name == "2026Q3"
    assert len(parts) == 11


def test_quarters_can_be_built_from_sparse_factor_calendar():
    calendar = pd.to_datetime(["2020-01-31", "2020-04-30", "2020-07-31"])
    parts = TimeSplitter(2020, "2018-01-01", "2020-12-31").quarters(calendar)
    assert [part.name for part in parts] == ["2020Q1", "2020Q2", "2020Q3"]


def test_monthly_frequency_refits_once_per_factor_month():
    calendar = pd.to_datetime(["2023-12-31", "2024-01-31", "2024-02-29", "2024-03-31"])

    parts = TimeSplitter(2024, "2024-01-01", "2024-03-31").rolling_periods(calendar, "monthly")

    assert [part.name for part in parts] == ["2024-01", "2024-02", "2024-03"]
    assert [part.train_cutoff for part in parts] == list(calendar[:3])
    assert [part.prediction_start for part in parts] == list(calendar[1:])


def test_window_validation_folds_only_use_current_rolling_window():
    dates = pd.date_range("2020-01-31", "2024-12-31", freq="ME")
    data = pd.DataFrame({
        "factor_date": dates,
        "label_exit_date": dates + pd.Timedelta(days=1),
        "label": np.arange(len(dates), dtype=float),
    })
    current_window = data[data.factor_date >= "2022-01-01"].copy()

    folds = RollingPredictor._window_validation_folds(current_window, 2)

    assert [name for name, _, _ in folds] == ["2024Q3", "2024Q4"]
    for _, train_index, valid_index in folds:
        assert data.loc[train_index, "factor_date"].min() >= pd.Timestamp("2022-01-01")
        assert data.loc[train_index, "label_exit_date"].max() < data.loc[valid_index, "factor_date"].min()
