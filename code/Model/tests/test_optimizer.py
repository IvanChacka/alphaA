from imports import *
import pytest

from optimizers.base import NoOptimizer
from optimizers.industry_neutral import IndustryNeutralOptimizer


def _scores_and_pool():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    columns = ["000001", "000002", "000003", "000004"]
    scores = pd.DataFrame([[1.0, 3.0, 2.0, 6.0], [4.0, 2.0, 5.0, 1.0]],
                          index=dates, columns=columns)
    pool = pd.DataFrame(True, index=dates, columns=columns)
    return scores, pool


def test_industry_neutral_scores_have_zero_group_mean_and_keep_ranking(tmp_path):
    path = tmp_path / "industry.csv"
    path.write_text("ticker,industry\n000001,bank\n000002,bank\n000003,tech\n000004,tech\n",
                    encoding="utf-8")
    scores, pool = _scores_and_pool()
    optimized, audit = IndustryNeutralOptimizer(path).transform(scores, pool)
    industries = pd.Series({"000001": "bank", "000002": "bank",
                            "000003": "tech", "000004": "tech"})
    for date in scores.index:
        means = optimized.loc[date].groupby(industries).mean()
        assert means.abs().max() < 1e-12
        for members in (["000001", "000002"], ["000003", "000004"]):
            assert optimized.loc[date, members].rank().equals(scores.loc[date, members].rank())
        assert np.isclose(optimized.loc[date].std(ddof=0), scores.loc[date].std(ddof=0))
    assert audit.max_abs_industry_mean.max() < 1e-12


def test_industry_neutral_rejects_missing_pool_coverage(tmp_path):
    path = tmp_path / "industry.csv"
    path.write_text("ticker,industry\n000001,bank\n", encoding="utf-8")
    scores, pool = _scores_and_pool()
    with pytest.raises(ValueError, match="行业覆盖缺失"):
        IndustryNeutralOptimizer(path).transform(scores, pool)


def test_dated_industry_mapping_never_backfills_future_information(tmp_path):
    path = tmp_path / "industry.csv"
    path.write_text(
        "date,ticker,industry\n2024-02-01,000001,bank\n2024-02-01,000002,bank\n"
        "2024-02-01,000003,tech\n2024-02-01,000004,tech\n", encoding="utf-8")
    scores, pool = _scores_and_pool()
    with pytest.raises(ValueError, match="行业覆盖缺失"):
        IndustryNeutralOptimizer(path).transform(scores, pool)


def test_no_optimizer_is_identity_copy():
    scores, pool = _scores_and_pool()
    optimized, audit = NoOptimizer().transform(scores, pool)
    pd.testing.assert_frame_equal(optimized, scores)
    assert optimized is not scores
    assert audit.empty
