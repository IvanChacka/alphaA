from imports import *
import pytest
from env import BACKTEST_DIR


def test_excess_metrics_use_the_relative_nav_curve():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from analytics import performance

    dates = pd.bdate_range("2024-01-02", periods=4)
    account = pd.DataFrame(
        {"total_asset": [100.0, 110.0, 99.0, 118.8]}, index=dates)
    benchmark = pd.Series([100.0, 105.0, 94.5, 103.95], index=dates)

    metrics, curve = performance(account, benchmark, annual_days=3)
    expected_excess_nav = curve.strategy / curve.benchmark
    expected_annual_return = expected_excess_nav.iloc[-1] / expected_excess_nav.iloc[0] - 1

    pd.testing.assert_series_equal(
        curve.excess, expected_excess_nav, check_names=False)
    assert metrics["excess_annual_return"] == pytest.approx(expected_annual_return)
    assert metrics["excess_max_drawdown"] == pytest.approx(
        (expected_excess_nav / expected_excess_nav.cummax() - 1).min())
