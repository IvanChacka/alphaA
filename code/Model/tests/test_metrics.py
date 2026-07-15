from imports import *
from rolling_ml.metrics import layered_metrics, metric_summary


def test_metrics_include_required_turnover_and_layers():
    dates = pd.to_datetime(["2024-01-02"] * 5 + ["2024-04-01"] * 5)
    frame = pd.DataFrame({"factor_date": dates, "symbol": [f"{x:06d}" for x in range(5)] * 2,
                          "prediction": list(range(5)) * 2, "label": list(range(5)) * 2,
                          "model": "linear_regression"})
    assert "prediction_turnover" in metric_summary(frame)
    output = layered_metrics(frame)
    assert {"full_test", "year", "quarter"}.issubset(set(output.level))
