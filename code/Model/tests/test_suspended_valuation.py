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


def test_a_share_fee_schedule_is_directional_and_date_aware():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    engine = BacktestEngine({}, BacktestConfig())
    assert np.isclose(engine._transaction_fee("2023-01-03", "BUY", 100_000), 31.0)
    assert np.isclose(engine._transaction_fee("2023-01-03", "SELL", 100_000), 131.0)
    assert np.isclose(engine._transaction_fee("2023-08-28", "SELL", 100_000), 81.0)
    assert np.isclose(engine._transaction_fee("2024-01-03", "BUY", 1_000), 5.01)
    legacy = BacktestEngine({}, BacktestConfig(fee_rate=.0014))
    assert np.isclose(legacy._transaction_fee("2024-01-03", "SELL", 100_000), 140.0)
    no_fee = BacktestEngine({}, BacktestConfig(fee_rate=0.0))
    assert no_fee._transaction_fee("2024-01-03", "SELL", 100_000) == 0.0


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


def test_rebalance_frequency_schedule_supports_four_modes():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    dates = pd.bdate_range("2024-01-02", periods=24)
    expected = {"daily": 23, "alternate": 12, "weekly": 5, "monthly": 2, "quarterly": 1}
    for frequency, count in expected.items():
        engine = BacktestEngine({}, BacktestConfig(rebalance_frequency=frequency))
        actual = sum(engine._scheduled_rebalance(i, dates) for i in range(1, len(dates)))
        assert actual == count


def test_decile_rotation_buys_top_decile_and_sells_worst_holding_decile():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    dates = pd.bdate_range("2024-01-02", periods=4)
    columns = [f"{index:06d}" for index in range(20)]
    prices = pd.DataFrame(10.0, index=dates, columns=columns)
    first = np.arange(20, dtype=float)[::-1]
    factors = pd.DataFrame([first, -first, first, first], index=dates, columns=columns)
    data = {"twap": prices, "close": prices.copy(),
            "adj": pd.DataFrame(1.0, index=dates, columns=columns),
            "factor": factors,
            "st": pd.DataFrame(False, index=dates, columns=columns),
            "pool": pd.DataFrame(True, index=dates, columns=columns),
            "benchmark": pd.Series(100.0, index=dates)}
    result = BacktestEngine(data, BacktestConfig(
        initial_cash=1_000_000, holding_count=10, max_turnover_ratio=1.0,
        selection_mode="decile_rotation", rotation_quantile=.10)).run()

    initial_buys = result.orders[(result.orders.date == dates[1]) &
                                 (result.orders.side == "BUY")]
    next_orders = result.orders[result.orders.date == dates[2]]
    assert len(initial_buys) == 10
    assert (next_orders.side == "SELL").sum() == 1
    assert (next_orders.side == "BUY").sum() == 1


def test_top_quantile_fully_reselects_without_turnover_limit():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    dates = pd.bdate_range("2024-01-30", periods=4)
    columns = [f"{index:06d}" for index in range(20)]
    prices = pd.DataFrame(10.0, index=dates, columns=columns)
    first = np.arange(20, dtype=float)[::-1]
    factors = pd.DataFrame([first, -first, first, first], index=dates, columns=columns)
    data = {"twap": prices, "close": prices.copy(),
            "adj": pd.DataFrame(1.0, index=dates, columns=columns),
            "factor": factors, "st": pd.DataFrame(False, index=dates, columns=columns),
            "pool": pd.DataFrame(True, index=dates, columns=columns),
            "benchmark": pd.Series(100.0, index=dates)}

    result = BacktestEngine(data, BacktestConfig(
        initial_cash=1_000_000, holding_count=10, max_turnover_ratio=.01,
        selection_mode="top_quantile", rotation_quantile=.10)).run()

    first_codes = set(result.holdings.loc[result.holdings.date == dates[1], "code"])
    second_codes = set(result.holdings.loc[result.holdings.date == dates[2], "code"])
    assert first_codes == {"000000", "000001"}
    assert second_codes == {"000018", "000019"}


def test_sparse_signal_eligibility_is_carried_between_signal_dates():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from data_loader import MarketData

    factor = pd.DataFrame({"a": [1.0, np.nan, np.nan],
                           "b": [2.0, np.nan, 3.0]})
    eligibility = MarketData._signal_eligibility(factor)

    assert eligibility.iloc[1].tolist() == [True, True]
    assert eligibility.iloc[2].tolist() == [False, True]


def test_sparse_signal_does_not_liquidate_between_rebalances():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    dates = pd.bdate_range("2024-01-02", periods=4)
    columns = ["000001", "000002"]
    prices = pd.DataFrame(10.0, index=dates, columns=columns)
    factors = pd.DataFrame([[2.0, 1.0], [np.nan, np.nan], [np.nan, np.nan],
                            [np.nan, np.nan]], index=dates, columns=columns)
    data = {"twap": prices, "close": prices.copy(),
            "adj": pd.DataFrame(1.0, index=dates, columns=columns),
            "factor": factors, "st": pd.DataFrame(False, index=dates, columns=columns),
            "pool": pd.DataFrame(True, index=dates, columns=columns),
            "benchmark": pd.Series(100.0, index=dates)}

    result = BacktestEngine(data, BacktestConfig(
        initial_cash=1_000_000, holding_count=1, selection_mode="top_quantile",
        rotation_quantile=.50, rebalance_frequency="daily", fee_rate=0.0)).run()

    assert len(result.orders) == 1
    assert result.account.holding_count.tolist() == [0, 1, 1, 1]
    assert result.risk_events.scheduled_rebalance.tolist() == [True, False, False]


def test_adjusted_research_prices_can_disable_false_limit_checks():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    date = pd.Timestamp("2024-01-03")
    code = "000001"
    data = {"twap": pd.DataFrame({code: [12.0]}, index=[date]),
            "adj": pd.DataFrame({code: [1.0]}, index=[date]),
            "st": pd.DataFrame({code: [False]}, index=[date])}
    previous = pd.Series({code: 10.0})

    strict = BacktestEngine(data, BacktestConfig(enforce_price_limits=True))
    research = BacktestEngine(data, BacktestConfig(enforce_price_limits=False))
    assert not strict._tradeable(date, code, "buy", previous)
    assert research._tradeable(date, code, "buy", previous)


def test_close_only_research_mode_liquidates_stale_position_at_last_close():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine, Position
    from config import BacktestConfig

    date = pd.Timestamp("2024-02-01")
    code = "000001"
    data = {"twap": pd.DataFrame({code: [np.nan]}, index=[date]),
            "adj": pd.DataFrame({code: [1.0]}, index=[date]),
            "st": pd.DataFrame({code: [False]}, index=[date])}
    engine = BacktestEngine(data, BacktestConfig(
        liquidate_missing_at_last_close=True, fee_rate=0.0))
    engine.cash = 0.0
    engine.positions[code] = Position(100.0, 9.0, pd.Timestamp("2024-01-02"))
    engine.last_close[code] = 10.0

    assert engine._sell(date, code, pd.Series({code: 10.0}))
    assert code not in engine.positions
    assert engine.cash == 1_000.0
    assert engine.orders[-1][4] == 10.0


def test_top_quantile_normalizes_remaining_cash_across_all_buyable_targets():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    dates = pd.bdate_range("2024-01-02", periods=4)
    columns = ["000001", "000002", "000003", "000004"]
    prices = pd.DataFrame(10.0, index=dates, columns=columns)
    # The first selected name becomes untradeable exactly when the next signal
    # requests names three and four.
    prices.loc[dates[2]:, "000001"] = np.nan
    factors = pd.DataFrame([[4.0, 3.0, 2.0, 1.0],
                            [1.0, 2.0, 4.0, 3.0],
                            [np.nan] * 4, [np.nan] * 4],
                           index=dates, columns=columns)
    data = {"twap": prices, "close": prices.copy(),
            "adj": pd.DataFrame(1.0, index=dates, columns=columns),
            "factor": factors, "st": pd.DataFrame(False, index=dates, columns=columns),
            "pool": pd.DataFrame(True, index=dates, columns=columns),
            "benchmark": pd.Series(100.0, index=dates)}

    result = BacktestEngine(data, BacktestConfig(
        initial_cash=1_000_000, selection_mode="top_quantile", rotation_quantile=.50,
        weight_mode="score", allow_fractional_shares=True,
        enforce_price_limits=False, fee_rate=0.0)).run()

    second_rebalance = set(result.holdings.loc[
        result.holdings.date == dates[2], "code"])
    assert second_rebalance == {"000001", "000003", "000004"}


def test_score_weights_use_rank_and_resist_scale_outliers():
    if str(BACKTEST_DIR) not in sys.path:
        sys.path.insert(0, str(BACKTEST_DIR))
    from backtest import BacktestEngine
    from config import BacktestConfig

    engine = BacktestEngine({}, BacktestConfig(weight_mode="score"))
    scores = pd.Series({"a": 1.0, "b": 2.0, "c": 1_000_000.0})
    weights = engine._allocation_weights(scores, ["a", "b", "c"])

    assert weights["a"] < weights["b"] < weights["c"]
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["c"] < 0.5


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
