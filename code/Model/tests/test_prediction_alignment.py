from imports import *
from rolling_ml.backtest_adapter import BacktestAdapter

def test_prediction_stays_on_factor_date():
    frame=pd.DataFrame({"factor_date":[pd.Timestamp("2024-01-02")],"symbol":["1"],"prediction":[.7]})
    wide=BacktestAdapter.prediction_wide(frame);assert wide.index[0]==pd.Timestamp("2024-01-02");assert wide.columns[0]=="000001"


def test_pool_icir_uses_daily_rank_ic_mean_over_std_without_annualization():
    values = [0.01, 0.03, 0.05]
    metrics = BacktestAdapter.pool_ic_statistics(values)
    expected = np.mean(values) / np.std(values, ddof=1)
    assert np.isclose(metrics["pool_icir"], expected)
    assert "pool_annualized_icir" not in metrics
