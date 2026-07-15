from imports import *
import pytest
from env import BACKTEST_DIR


def test_consecutive_suspension_keeps_last_valid_close():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    dates = pd.bdate_range("2024-01-02", periods=4)
    columns = ["000001"]
    frame = lambda values: pd.DataFrame({"000001": values}, index=dates)
    data = {"twap": frame([10.0, 10.0, np.nan, np.nan]),
            "close": frame([10.0, 10.0, np.nan, np.nan]),
            "adj": frame([1.0, 1.0, np.nan, np.nan]),
            "factor": frame([1.0, 1.0, 1.0, 1.0]),
            "st": frame([False] * 4), "pool": frame([True] * 4),
            "benchmark": pd.Series([100.0] * 4, index=dates)}
    result = BacktestEngine(data, BacktestConfig(initial_cash=1_000_000, holding_count=1,
                                                  max_turnover_ratio=1.0)).run()
    assert np.isfinite(result.account.total_asset).all()
    assert result.holdings.loc[result.holdings.date == dates[-1], "close"].iloc[0] == 10.0


def test_performance_rejects_nonfinite_account():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from analytics import performance
    dates = pd.bdate_range("2024-01-02", periods=2)
    account = pd.DataFrame({"total_asset": [1.0, np.nan]}, index=dates)
    with pytest.raises(ValueError, match="非有限值"):
        performance(account, pd.Series([1.0, 1.0], index=dates))
