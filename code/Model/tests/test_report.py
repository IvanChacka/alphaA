from imports import *
import pytest
from rolling_ml.report_generator import ReportGenerator

pytestmark = pytest.mark.filterwarnings("ignore:.*scattermapbox.*:DeprecationWarning")


def test_report_is_self_contained_and_contains_chinese(tmp_path):
    path = tmp_path / "index.html"
    ReportGenerator().generate(path, "test_run", {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    text = path.read_text(encoding="utf-8")
    assert "滚动机器学习样本外报告" in text
    assert "<script src=" not in text
    assert path.stat().st_size > 1_000_000
