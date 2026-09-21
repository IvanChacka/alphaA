from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_framework.analysis import ICAnalyzer, LayeredBacktester
from research_framework.data import DataSourceFactory, LocalParquetSource, TushareSource
from research_framework.factors import Factor, FactorPipeline, FactorRegistry
from research_framework.models import framework_manifest
from research_framework.portfolio import PortfolioConfig, PortfolioEngine, performance_metrics
from research_framework.service import ResearchService


def test_parquet_source_detects_index_columns_and_coverage(tmp_path):
    path = tmp_path / "factor.parquet"
    frame = pd.DataFrame({"date": [20240102, 20240103], "ticker": [1, 2],
                          "value": [.1, .2]}).set_index(["date", "ticker"])
    frame.to_parquet(path)
    source = DataSourceFactory.create(path)
    profile = source.inspect()
    loaded = source.load("2024-01-02", "2024-01-03", ["date", "ticker", "value"])
    assert isinstance(source, LocalParquetSource)
    assert (profile.start, profile.end) == ("2024-01-02", "2024-01-03")
    assert {"date", "ticker", "value"}.issubset(loaded.columns)


def test_tushare_uri_is_recognized_without_network_call():
    source = DataSourceFactory.create("tushare://daily")
    assert isinstance(source, TushareSource)
    assert source.inspect().provider == "tushare"


def test_factor_registry_skips_missing_input_and_cross_sectionally_standardizes():
    registry = FactorRegistry()

    @registry.register
    class Demo(Factor):
        name = "demo"
        required_columns = ("raw",)
        def compute(self, data):
            return data.raw

    data = pd.DataFrame({"date": pd.to_datetime(["2024-01-02"] * 3),
                         "symbol": ["1", "2", "3"], "raw": [1., 2., 3.]})
    with pytest.warns(RuntimeWarning, match="因子 missing 已跳过"):
        result = FactorPipeline(registry).run(data, [
            {"name": "demo", "output": "factor"},
            {"name": "demo", "output": "missing", "parameters": {"unused": 1}},
        ])
    assert np.isclose(result.values.factor.mean(), 0)
    assert np.isclose(result.values.factor.std(ddof=0), 1)
    assert result.skipped == ["missing"]


def test_ic_uses_first_session_after_signal_then_full_holding_period():
    dates = pd.bdate_range("2024-01-02", periods=5)
    prices = pd.DataFrame({"a": [10, 11, 13, 14, 15]}, index=dates)
    signal_date = pd.DatetimeIndex([dates[0]])
    result = ICAnalyzer.returns_for_signal_dates(prices, signal_date, 1)
    assert result.loc[dates[0], "a"] == 13 / 11 - 1


def test_layer_backtest_outputs_deciles_and_long_short():
    dates = pd.bdate_range("2024-01-02", periods=5)
    symbols = [str(i) for i in range(20)]
    factor = pd.DataFrame([np.arange(20)], index=dates[:1], columns=symbols)
    prices = pd.DataFrame([np.ones(20), np.ones(20), np.linspace(.9, 1.2, 20),
                           np.ones(20), np.ones(20)], index=dates, columns=symbols)
    result = LayeredBacktester(10, "D").run(factor, prices)
    assert len(result["summary"]) == 10
    assert result["long_short"]


def test_portfolio_engine_applies_costs_and_shared_performance_metrics():
    dates = pd.bdate_range("2024-01-02", periods=3)
    scores = pd.DataFrame({"a": [2, 2, 2], "b": [1, 1, 1]}, index=dates)
    returns = pd.DataFrame({"a": [.01, .01, .01], "b": [0, 0, 0]}, index=dates)
    result = PortfolioEngine(PortfolioConfig(rebalance_frequency="D", top_n=1)).run(scores, returns)
    assert result["total_cost"] > 0
    assert result["metrics"]["sharpe"] == performance_metrics(
        result["curve"].net_return)["sharpe"]


def test_framework_manifest_exposes_models_and_all_research_layers():
    manifest = framework_manifest()
    assert len(manifest["architecture"]) == 5
    assert set(manifest["models"]) == {
        "linear_regression", "ridge", "elasticnet", "xgboost", "style_rotation"}
    assert manifest["ic_horizons"] == [5, 10, 20, 60]


def test_framework_metadata_exposes_factor_catalog_with_stable_ordinals(tmp_path):
    factors = tmp_path / "factors.parquet"
    market = tmp_path / "market.parquet"
    pd.DataFrame({"date": [20240102], "ticker": ["000001"],
                  "alpha_a": [.1], "alpha_b": [.2]}).to_parquet(factors)
    pd.DataFrame({"TradingDate": [pd.Timestamp("2024-01-02")],
                  "Stkcd": ["000001"], "price": [10.]}).to_parquet(market)
    metadata = ResearchService({"demo": factors}, market, tmp_path / "uploads").metadata()
    source = next(item for item in metadata["data_sources"] if item["id"] == "demo")
    assert source["factor_catalog"] == [
        {"index": 1, "name": "alpha_a", "expression": None,
         "description": "原文件未提供表达式", "source": "demo"},
        {"index": 2, "name": "alpha_b", "expression": None,
         "description": "原文件未提供表达式", "source": "demo"},
    ]
