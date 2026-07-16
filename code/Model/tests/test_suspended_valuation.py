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


def test_trade_gate_freezes_rebalance_without_forced_liquidation():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    dates = pd.bdate_range("2024-01-02", periods=4)
    columns = ["000001", "000002"]
    prices = pd.DataFrame(10.0, index=dates, columns=columns)
    factors = pd.DataFrame([[2.0, 1.0], [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
                           index=dates, columns=columns)
    data = {
        "twap": prices.copy(), "close": prices.copy(),
        "adj": pd.DataFrame(1.0, index=dates, columns=columns),
        "factor": factors,
        "st": pd.DataFrame(False, index=dates, columns=columns),
        "pool": pd.DataFrame(True, index=dates, columns=columns),
        "benchmark": pd.Series(100.0, index=dates),
        # t0允许首次建仓；t1信号会作用于t2并暂停换仓。
        "trade_enabled": pd.Series([True, False, True, True], index=dates),
    }
    result = BacktestEngine(data, BacktestConfig(
        initial_cash=1_000_000, holding_count=1, max_turnover_ratio=1.0)).run()
    day_two_orders = result.orders[result.orders.date == dates[2]]
    assert day_two_orders.empty
    assert set(result.holdings.loc[result.holdings.date == dates[2], "code"]) == {"000001"}
    event = result.risk_events[result.risk_events.date == dates[2]].iloc[0]
    assert not event.signal_enabled
    assert not event.trade_enabled


def test_quadratic_drawdown_optimizer_reduces_exposure_without_changing_stock():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig
    from optimizers.drawdown import QuadraticDrawdownOptimizer

    dates = pd.bdate_range("2024-01-02", periods=30)
    columns = ["000001"]
    prices = pd.DataFrame(
        10.0 * np.power(.99, np.arange(len(dates)))[:, None],
        index=dates, columns=columns)
    optimizer = QuadraticDrawdownOptimizer(.10, min_observations=5)
    data = {
        "twap": prices.copy(), "close": prices.copy(),
        "adj": pd.DataFrame(1.0, index=dates, columns=columns),
        "factor": pd.DataFrame(1.0, index=dates, columns=columns),
        "st": pd.DataFrame(False, index=dates, columns=columns),
        "pool": pd.DataFrame(True, index=dates, columns=columns),
        "benchmark": pd.Series(100.0, index=dates),
        "risk_off": pd.Series(False, index=dates),
        "drawdown_optimizer": optimizer,
    }
    result = BacktestEngine(data, BacktestConfig(
        initial_cash=1_000_000, holding_count=1, max_turnover_ratio=1.0)).run()
    assert result.risk_events.target_exposure.min() < 1.0
    assert result.risk_events.actual_exposure.min() < .95
    assert result.account.holding_count.iloc[-1] == 1
    assert (result.orders.side == "SELL").any()
