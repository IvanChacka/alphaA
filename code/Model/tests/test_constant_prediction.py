from imports import *
from env import MLConfig
from rolling_ml.experiment_logger import ExperimentLogger
from rolling_ml.metrics import daily_ic
from rolling_ml.rolling_predict import RollingPredictor


def test_daily_ic_handles_constant_prediction_without_warning():
    frame = pd.DataFrame({"factor_date": pd.to_datetime(["2024-01-02"] * 4),
                          "prediction": [1.0] * 4, "label": [0.1, 0.2, -0.1, 0.0]})
    with warnings.catch_warnings(record=True) as caught:
        result = daily_ic(frame)
    assert pd.isna(result.loc[0, "rank_ic"])
    assert not caught


def test_elasticnet_rejects_constant_candidates(tmp_path):
    dates = np.repeat(pd.date_range("2022-01-01", periods=4), 20)
    alpha = np.tile(np.linspace(-1, 1, 20), 4)
    data = pd.DataFrame({"factor_date": dates, "symbol": [f"{i:06d}" for i in range(20)] * 4,
                         "label": alpha + np.sin(np.arange(80)) * .01, "alpha": alpha})
    folds = [("fold", data.index[:60], data.index[60:])]
    logger = ExperimentLogger(tmp_path)
    try:
        predictor = RollingPredictor(MLConfig(models=("elasticnet",)), logger)
        predictor._linear_params("elasticnet", data, ["alpha"], folds)
        progress = pd.read_csv(logger.root / "metrics/elasticnet_tuning_progress.csv")
        assert "rejected" in set(progress.state)
        assert "constant_prediction" in set(progress.reason.dropna())
    finally:
        logger.close()
