from imports import *
import pytest
from rolling_ml.report_generator import ReportGenerator

pytestmark = pytest.mark.filterwarnings("ignore:.*scattermapbox.*:DeprecationWarning")


def test_report_is_self_contained_and_contains_chinese(tmp_path):
    path = tmp_path / "index.html"
    training = pd.DataFrame({
        "model": ["style_rotation"], "quarter": ["2024Q1"],
        "train_rank_ic_5": [.2], "oos_style_rank_ic_5": [.1],
        "train_rank_ic_20": [.3], "oos_style_rank_ic_20": [.15],
    })
    pca_styles = pd.DataFrame({
        "quarter": ["2024Q1"], "style": ["style_1"],
        "mean_rank_ic": [.04]})
    ReportGenerator().generate(
        path, "test_run", {}, pd.DataFrame(), pd.DataFrame(), training,
        pca_style_metrics=pca_styles)
    text = path.read_text(encoding="utf-8")
    assert "滚动机器学习样本外报告" in text
    assert "PCA style_1 股票RankIC" in text
    assert "固定20日隔离验证（单点）" not in text
    assert "<script src=" not in text
    assert path.stat().st_size > 1_000_000
