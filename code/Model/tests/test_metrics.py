from imports import *
from rolling_ml.metrics import daily_ic, layered_metrics, metric_summary, model_selection_score
from rolling_ml.backtest_adapter import BacktestAdapter
import sys
sys.path.insert(0, str(Path(__file__).parents[2] / "BackTest"))
from analytics import performance


def test_daily_ic_empty_frame_keeps_metric_schema():
    result = daily_ic(pd.DataFrame(columns=["factor_date", "prediction", "label"]))
    assert result.empty
    assert "rank_ic" in result.columns


def test_metric_summary_empty_frame_is_safe():
    result = metric_summary(pd.DataFrame(columns=["factor_date", "prediction", "label"]))
    assert result["observation_days"] == 0


def test_model_selection_score_combines_rank_pearson_and_returns():
    positive = model_selection_score({"mean_rank_ic": .1, "pearson_ic": .1,
                                      "top_bottom_return": .05})
    negative = model_selection_score({"mean_rank_ic": .1, "pearson_ic": .1,
                                      "top_bottom_return": -.05})
    assert positive > negative


def test_performance_annualizes_monthly_nav_by_elapsed_years():
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")
    account = pd.DataFrame({"total_asset": np.linspace(100, 200, len(dates))}, index=dates)
    benchmark = pd.Series(np.linspace(100, 200, len(dates)), index=dates)
    metrics, _ = performance(account, benchmark)
    assert 0.04 < metrics["absolute_annual_return"] < 0.10


def test_metrics_include_required_turnover_and_layers():
    dates = pd.to_datetime(["2024-01-02"] * 5 + ["2024-04-01"] * 5)
    frame = pd.DataFrame({"factor_date": dates, "symbol": [f"{x:06d}" for x in range(5)] * 2,
                          "prediction": list(range(5)) * 2, "label": list(range(5)) * 2,
                          "model": "linear_regression"})
    assert "prediction_turnover" in metric_summary(frame)
    output = layered_metrics(frame)
    assert {"full_test", "year", "quarter"}.issubset(set(output.level))


def test_score_layer_curve_builds_decile_navs_and_long_short():
    dates = pd.to_datetime(["2024-01-02"] * 10 + ["2024-02-01"] * 10)
    frame = pd.DataFrame({"factor_date": dates,
                          "symbol": [f"{x:06d}" for x in range(10)] * 2,
                          "prediction": list(range(10)) * 2,
                          "label": [x / 100 for x in range(10)] * 2})
    layers = BacktestAdapter.score_layer_curve(frame)
    assert len(layers) == 2
    assert {"q1", "q10", "q10_q1"}.issubset(layers.columns)
    assert layers.q10.iloc[-1] > layers.q1.iloc[-1]
