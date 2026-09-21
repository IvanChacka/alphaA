from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from style_rotation.pca_style import FixedPCAStyle, varimax
from style_rotation.pipeline import StyleRotationPipeline
from style_rotation.config import StyleRuntimeConfig
from style_rotation.models import DualHorizonXGBoost, PCAStockXGBoost, PCAStockXGBoostTuner
from style_rotation.residual_model import ResidualRidge
from style_rotation.style_returns import StyleReturnBuilder
from style_rotation.timing import TimingDatasetBuilder


def test_style_runtime_config_accepts_auditable_xgb_depth():
    assert StyleRuntimeConfig(xgb_max_depth=4).validate().xgb_max_depth == 4
    assert StyleRuntimeConfig(rotation_quantile=.10,
                              rebalance_frequency="monthly").validate().rebalance_frequency == "monthly"
    assert StyleRuntimeConfig(rebalance_frequency="quarterly").validate().rebalance_frequency == "quarterly"
    with pytest.raises(ValueError, match="调仓频率"):
        StyleRuntimeConfig(rebalance_frequency="yearly").validate()


def test_pca_stock_xgboost_uses_exposures_and_purged_validation(monkeypatch):
    monkeypatch.setattr("style_rotation.models.STYLE_XGB_MAX_ESTIMATORS", 12)
    monkeypatch.setattr("style_rotation.models.STYLE_XGB_EARLY_STOPPING_ROUNDS", 3)
    monkeypatch.setattr("style_rotation.models.STYLE_XGB_VALIDATION_EMBARGO_DAYS", 4)
    rng = np.random.default_rng(71)
    rows = []
    for date in pd.date_range("2022-01-03", periods=40, freq="B"):
        for stock in range(12):
            values = rng.normal(size=3)
            rows.append({"factor_date": date, "symbol": f"{stock:06d}",
                         "style_1": values[0], "style_2": values[1],
                         "style_3": values[2],
                         "label": values[0] - .4 * values[1] + rng.normal(scale=.1),
                         "sample_weight": 1.0})
    frame = pd.DataFrame(rows)
    frame.loc[0, "label"] = np.nan
    model = PCAStockXGBoost(
        ["style_1", "style_2", "style_3"], {"max_depth": 2}).fit(frame)

    assert len(model.predict(frame.head(7))) == 7
    assert set(model.feature_importance_frame().feature) == {
        "style_1", "style_2", "style_3"}
    assert model.best_trees >= 1
    assert model.fit_end_ < model.validation_start_
    business_gap = len(pd.bdate_range(model.fit_end_, model.validation_start_)) - 1
    assert business_gap >= model.embargo_days_


def test_pca_xgb_optuna_split_drops_nonfinite_labels():
    dates = pd.date_range("2022-01-03", periods=30, freq="B")
    frame = pd.DataFrame({"factor_date": dates, "symbol": ["000001"] * len(dates),
                          "style_1": 1.0, "label": 0.1, "sample_weight": 1.0})
    frame.loc[0, "label"] = np.nan

    fit, valid = PCAStockXGBoostTuner._split(frame, embargo_days=4)

    assert fit.label.notna().all() and valid.label.notna().all()


def test_pca_xgb_default_split_has_no_fixed_embargo_and_purges_by_label_exit():
    dates = pd.date_range("2022-01-31", periods=5, freq="ME")
    frame = pd.DataFrame({"factor_date": dates, "symbol": "000001",
                          "style_1": 1.0, "label": np.arange(len(dates)),
                          "sample_weight": 1.0,
                          "label_exit_date": dates.to_series().shift(-1).to_numpy()})

    fit, valid = PCAStockXGBoostTuner._split(frame)

    assert valid.factor_date.min() == dates[4]
    assert fit.factor_date.max() == dates[2]
    assert (fit.label_exit_date < valid.factor_date.min()).all()


def test_pca_stock_xgb_uses_no_unrequested_fixed_embargo():
    assert PCAStockXGBoost(["style_1"]).embargo_days_ == 0


def test_pca_xgb_converts_twenty_trading_day_embargo_for_monthly_factors():
    dates = pd.date_range("2009-04-30", periods=9, freq="ME")
    frame = pd.DataFrame({"factor_date": dates, "symbol": "000001",
                          "style_1": 1.0, "label": np.arange(len(dates)),
                          "sample_weight": 1.0})

    fit, valid = PCAStockXGBoostTuner._split(frame, embargo_days=20)

    omitted = dates[(dates > fit.factor_date.max()) &
                    (dates < valid.factor_date.min())]
    assert len(omitted) == 1
    assert len(valid.factor_date.unique()) >= 1


def test_pca_confidence_gate_holds_then_liquidates_after_two_bad_windows():
    dates = pd.date_range("2024-01-02", periods=25, freq="B")
    rows = []
    for position, date in enumerate(dates):
        exit_date = dates[min(position + 1, len(dates) - 1)]
        for value, symbol in enumerate(["000001", "000002", "000003"], 1):
            rows.append({"factor_date": date, "symbol": symbol,
                         "prediction": float(value), "label": float(4 - value),
                         "label_exit_date": exit_date, "pool": "A500"})
    predictions = pd.DataFrame(rows)
    training = pd.DataFrame([{"pool": "A500", "quarter": "2024Q1",
                              "validation_rank_ic": .10}])
    metadata = {pd.Period("2024Q1", freq="Q"): {
        "cumulative_explained_variance": .95,
        "alignment": [{"signed_similarity": .90}]}}
    config = StyleRuntimeConfig(confidence_rankic_window=10, entry_rank=2,
                                exit_rank=3)
    gated, audit = StyleRotationPipeline(None, config=config)._apply_pca_confidence_gate(
        predictions, training, metadata)

    pd.testing.assert_series_equal(gated.prediction, predictions.prediction)
    assert set(audit["mode"]) == {"normal", "hold", "risk_off_liquidate"}
    hold_date = audit.loc[audit["mode"] == "hold", "factor_date"].iloc[0]
    held = gated[gated.factor_date == hold_date]
    assert held.trade_prediction.isna().all() and held.retain.all()
    risk_date = audit.loc[audit["mode"] == "risk_off_liquidate", "factor_date"].iloc[0]
    risk = gated[gated.factor_date == risk_date]
    assert risk.trade_prediction.isna().all() and not risk.retain.any()


def test_varimax_is_orthogonal():
    raw = np.random.default_rng(7).normal(size=(30, 6))
    _, rotation, _ = varimax(raw)
    assert np.allclose(rotation.T @ rotation, np.eye(6), atol=1e-6)


def test_pca_alignment_reorders_and_signs_components():
    reference = FixedPCAStyle()
    current = FixedPCAStyle()
    reference.features = current.features = ["f0", "f1", "f2"]
    reference.loadings = np.eye(3)
    current.loadings = reference.loadings[:, [2, 0, 1]] * np.array([-1.0, 1.0, -1.0])
    reference.explained_variance = np.array([.5, .3, .2])
    current.explained_variance = np.array([.2, .5, .3])
    reference.rotation = current.rotation = np.eye(3)

    mapping = current.align_to(reference)

    assert np.allclose(current.loadings, reference.loadings)
    assert mapping.source_component.tolist() == [2, 3, 1]


def test_fixed_pca_transform_does_not_refit():
    rng = np.random.default_rng(8)
    rows = []
    for date in pd.date_range("2020-01-01", periods=8, freq="B"):
        for symbol in range(50):
            rows.append([date, f"{symbol:06d}", *rng.normal(size=10)])
    columns = ["factor_date", "symbol"] + [f"f{i}" for i in range(10)]
    frame = pd.DataFrame(rows, columns=columns)
    model = FixedPCAStyle().fit(frame, columns[2:])
    before = model.loadings.copy()
    model.transform(frame.assign(f0=frame.f0 * 100))
    assert np.array_equal(before, model.loadings)
    assert len(model.style_names) >= 5
    assert model.cumulative_explained_variance >= .90


def test_fixed_pca_accepts_zero_variance_missing_factor():
    rng = np.random.default_rng(18)
    rows = []
    for date in pd.date_range("2020-01-01", periods=6, freq="B"):
        for symbol in range(40):
            rows.append([date, f"{symbol:06d}", *rng.normal(size=9), 0.0])
    columns = ["factor_date", "symbol"] + [f"f{i}" for i in range(10)]
    frame = pd.DataFrame(rows, columns=columns)
    model = FixedPCAStyle().fit(frame, columns[2:])
    assert np.isfinite(model.loadings).all()
    assert np.allclose(model.loadings[-1], 0.0)


def test_pca_style_rankic_reports_every_component_by_quarter():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    symbols = ["000001", "000002", "000003"]
    rows, labels = [], []
    for date in dates:
        for value, symbol in enumerate(symbols, start=1):
            rows.append({"factor_date": date, "symbol": symbol,
                         "style_1": float(value), "style_2": float(-value)})
            labels.append({"factor_date": date, "symbol": symbol,
                           "label": float(value)})
    quarter = pd.Period("2024Q1", freq="Q")
    daily, summary = StyleRotationPipeline._pca_style_rankic(
        {quarter: {"exposure": pd.DataFrame(rows)}},
        pd.DataFrame(labels), ["style_1", "style_2"])

    assert set(daily["style"]) == {"style_1", "style_2"}
    by_style = summary.set_index("style")
    assert np.isclose(by_style.loc["style_1", "mean_rank_ic"], 1.0)
    assert np.isclose(by_style.loc["style_2", "mean_rank_ic"], -1.0)
    assert set(summary.quarter) == {"2024Q1"}


def test_timing_features_only_use_realized_returns():
    dates = pd.date_range("2024-01-02", periods=12, freq="B")
    rows = []
    for i, date in enumerate(dates[:-2]):
        for style in ("style_1", "style_2"):
            rows.append({"factor_date": date, "style": style, "style_return": i / 100,
                         "style_entry_date": dates[i + 1], "style_exit_date": dates[i + 2]})
    features = TimingDatasetBuilder().build_features(pd.DataFrame(rows), dates)
    valid = features.dropna(subset=["max_information_date"])
    assert (valid.max_information_date <= valid.decision_date).all()
    availability = features.max_information_date.isna() | (features.max_information_date <= features.decision_date)
    assert availability.all()


def test_stock_score_keeps_factor_date():
    exposure = pd.DataFrame({"factor_date": [pd.Timestamp("2024-01-02")] * 3,
        "symbol": ["000001", "000002", "000003"], "style_1": [-1., 0., 1.], "style_2": [1., 0., -1.]})
    weights = pd.DataFrame({"decision_date": [pd.Timestamp("2024-01-02")], "style_1": [.8], "style_2": [.2]})
    result = StyleRotationPipeline._stock_scores(exposure, weights, ["style_1", "style_2"])
    assert result.factor_date.eq(pd.Timestamp("2024-01-02")).all()
    assert result.prediction.iloc[-1] > result.prediction.iloc[0]


def test_style_return_builder_emits_one_row_per_date_and_style():
    date = pd.Timestamp("2024-01-02")
    exposure = pd.DataFrame({"factor_date": [date] * 20,
        "symbol": [f"{i:06d}" for i in range(20)], "style_1": np.arange(20, dtype=float)})
    labels = pd.DataFrame({"factor_date": [date] * 20,
        "symbol": [f"{i:06d}" for i in range(20)], "label": np.arange(20, dtype=float) / 100,
        "label_entry_date": [pd.Timestamp("2024-01-03")] * 20,
        "label_exit_date": [pd.Timestamp("2024-01-04")] * 20})
    result = StyleReturnBuilder().build(exposure, labels, ["style_1"])
    assert len(result) == 1
    assert result.iloc[0].style == "style_1"
    assert result.iloc[0].style_return > 0


def test_twenty_day_label_has_later_information_date():
    dates = pd.date_range("2024-01-02", periods=25, freq="B")
    returns = pd.DataFrame([{"factor_date": date, "style": style, "style_return": .001,
        "style_entry_date": dates[min(i + 1, 24)], "style_exit_date": dates[min(i + 2, 24)]}
        for i, date in enumerate(dates) for style in ("style_1", "style_2")])
    labels = TimingDatasetBuilder().build_horizon_labels(returns, dates, 20)
    assert labels.iloc[0].label_exit_date_20 == dates[21]
    assert labels.iloc[0].label_exit_date_20 > labels.iloc[0].decision_date


def test_consensus_scores_keep_neutral_out_and_require_multiple_bearish_votes_to_sell():
    date = pd.Timestamp("2024-01-02")
    styles = [f"style_{i}" for i in range(1, 4)]
    exposure = pd.DataFrame({"factor_date": [date] * 10, "symbol": [f"{i:06d}" for i in range(10)]})
    for i, style in enumerate(styles): exposure[style] = np.linspace(-1, 1, 10) * (1 if i < 2 else -1)
    forecast = pd.DataFrame({"decision_date": [date], **{style: [1.0] for style in styles}})
    score, votes = StyleRotationPipeline._consensus_stock_scores(exposure, forecast, styles)
    assert score.prediction.notna().sum() < len(score)
    assert "retain" not in score
    assert votes.sell_votes.max() >= 2


def test_any_single_bullish_style_is_enough_to_buy():
    date = pd.Timestamp("2024-01-02")
    styles = ["style_1", "style_2", "style_3"]
    exposure = pd.DataFrame({"factor_date": [date] * 5,
        "symbol": [f"{i:06d}" for i in range(5)],
        "style_1": [-2., -1., 0., 1., 2.], "style_2": 0., "style_3": 0.})
    forecast = pd.DataFrame({"decision_date": [date], "style_1": [1.],
                             "style_2": [0.], "style_3": [0.]})
    score, votes = StyleRotationPipeline._consensus_stock_scores(
        exposure, forecast, styles, buy_confirmations=1, vote_quantile=.80)
    assert votes.buy_eligible.sum() == 1
    assert score.loc[votes.buy_eligible, "prediction"].notna().all()


def test_residual_projection_is_outside_pca_space_and_formula_adds_incremental_return():
    rng = np.random.default_rng(27)
    loadings = rng.normal(size=(6, 3))
    model = ResidualRidge(loadings, alpha=.1)
    assert np.allclose(model.residual_projection @ loadings, 0.0, atol=1e-10)
    style = pd.Series([1.0, .5])
    residual = pd.Series([.2, -.3])
    combined = StyleRotationPipeline._combine_style_residual(style, residual, 1.5)
    assert np.allclose(combined, [1.3, .05])


def test_component_ic_uses_same_final_candidate_universe():
    date = pd.Timestamp("2024-01-02")
    frame = pd.DataFrame({
        "factor_date": [date] * 4,
        "symbol": ["000001", "000002", "000003", "000004"],
        "style_timing_score": [1.0, 2.0, 3.0, np.nan],
        "residual_prediction_score": [3.0, 2.0, 1.0, 99.0],
        "prediction": [2.5, 3.0, 3.5, np.nan],
        "label": [1.0, 2.0, 3.0, -10.0],
        "model": "style_rotation",
    })
    metrics = StyleRotationPipeline._component_daily_metrics(frame)
    comparable = metrics[metrics.component.isin(
        ["style_timing", "orthogonal_residual", "final"])]
    assert comparable.stock_count.eq(3).all()
    all_market = metrics[metrics.component == "orthogonal_residual_all_universe"]
    assert all_market.stock_count.eq(4).all()


def test_point_in_time_pool_filter_uses_each_dates_membership():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    frame = pd.DataFrame({
        "factor_date": [dates[0], dates[0], dates[1], dates[1]],
        "symbol": ["000001", "000002", "000001", "000002"],
        "factor": [1.0, 2.0, 3.0, 4.0],
    })
    membership = pd.DataFrame(
        [[True, False], [False, True]], index=dates, columns=["000001", "000002"])

    selected = StyleRotationPipeline._inside_pool(frame, membership)

    assert list(zip(selected.factor_date, selected.symbol)) == [
        (dates[0], "000001"), (dates[1], "000002")]


def test_residual_target_is_orthogonal_to_style_exposures():
    rng = np.random.default_rng(31)
    rows = 100
    frame = pd.DataFrame({"factor_date": pd.Timestamp("2023-01-03"),
                          "style_1": rng.normal(size=rows),
                          "style_2": rng.normal(size=rows)})
    frame["label"] = 2 * frame.style_1 - frame.style_2 + rng.normal(scale=.1, size=rows)
    residual, audit = StyleRotationPipeline._build_residual_target(frame, ["style_1", "style_2"])
    assert abs(residual.corr(frame.style_1)) < 1e-10
    assert abs(residual.corr(frame.style_2)) < 1e-10
    assert audit.max_abs_style_correlation.iloc[0] < 1e-10


def test_residual_ridge_records_realized_label_boundary():
    rng = np.random.default_rng(29)
    features = [f"f{i}" for i in range(4)]
    frame = pd.DataFrame(rng.normal(size=(100, 4)), columns=features)
    frame["label"] = frame.f0 - frame.f1
    frame["label_exit_date"] = pd.date_range("2023-01-01", periods=100, freq="D")
    model = ResidualRidge(np.eye(4)[:, :2], alpha=.01).fit(frame, features)
    assert model.train_rows_ == 100
    assert model.max_label_exit_date_ == frame.label_exit_date.max()


def test_residual_validation_disables_regime_reversed_increment():
    rng = np.random.default_rng(71)
    dates = pd.date_range("2020-01-02", periods=100, freq="B")
    features = [f"f{i}" for i in range(4)]
    rows = []
    for position, date in enumerate(dates):
        values = rng.normal(size=(20, len(features)))
        direction = 1.0 if position < 80 else -1.0
        for symbol, vector in enumerate(values):
            rows.append({"factor_date": date, "symbol": f"{symbol:06d}",
                         **dict(zip(features, vector)),
                         "residual_label": direction * vector[1],
                         "sample_weight": 1.0})
    frame = pd.DataFrame(rows)
    pca = FixedPCAStyle()
    pca.features = features
    pca.loadings = np.eye(len(features))[:, :1]
    pipeline = StyleRotationPipeline(None, config=StyleRuntimeConfig(residual_weight=.8))

    result = pipeline._validated_residual_weight(frame, pca, features)

    assert result["rank_ic"] < 0
    assert result["weight"] == 0.0


def test_training_forecast_ic_is_reported():
    dates = pd.date_range("2023-01-02", periods=4, freq="B")
    styles = ["style_1", "style_2", "style_3"]
    forecast = pd.DataFrame({"decision_date": dates, "style_1": [1, 2, 3, 4],
                             "style_2": [2, 3, 4, 5], "style_3": [3, 4, 5, 6]})
    actual = pd.DataFrame({"decision_date": dates,
        "target_style_1_5": [1, 2, 3, 4], "target_style_2_5": [2, 3, 4, 5],
        "target_style_3_5": [3, 4, 5, 6]})
    assert np.isclose(StyleRotationPipeline._mean_forecast_ic(forecast, actual, styles, 5), 1.0)


def test_style_xgb_validation_has_twenty_trading_day_embargo():
    dates = pd.date_range("2022-01-03", periods=100, freq="B")
    frame = pd.DataFrame({"decision_date": dates})
    fit, valid = DualHorizonXGBoost._purged_time_split(frame, embargo_days=20)
    assert valid.decision_date.min() == dates[80]
    assert fit.decision_date.max() == dates[59]
    omitted = dates[(dates > fit.decision_date.max()) & (dates < valid.decision_date.min())]
    assert len(omitted) == 20


def test_style_xgb_converts_embargo_for_monthly_observations():
    dates = pd.date_range("2022-01-31", periods=12, freq="ME")
    frame = pd.DataFrame({"decision_date": dates})
    fit, valid = DualHorizonXGBoost._purged_time_split(frame, embargo_days=20)
    omitted = dates[(dates > fit.decision_date.max()) &
                    (dates < valid.decision_date.min())]
    assert len(omitted) == 1


def test_style_xgb_uses_one_shared_long_table_for_all_target_styles():
    styles = ["style_1", "style_2", "style_3"]
    model = DualHorizonXGBoost(styles, 5)
    dates = pd.date_range("2023-01-02", periods=2, freq="B")
    frame = pd.DataFrame({"decision_date": dates, "sample_weight": [1.0, .5]})
    for index, style in enumerate(styles):
        frame[f"{style}_momentum_5"] = index + 1.0
        frame[f"{style}_momentum_20"] = index + 2.0
        frame[f"{style}_volatility_5"] = index + 3.0
        frame[f"target_{style}_5"] = [index / 100, (index + 1) / 100]
    frame["benchmark_volatility_20"] = .02
    frame["benchmark_return_20"] = .03
    x, y, weights = model._long_matrix(frame, include_target=True)
    assert x.shape == (len(frame) * len(styles), len(model.model_feature_columns))
    assert np.allclose(x[model.style_indicator_columns].sum(axis=1), 1.0)
    assert np.allclose(y.reshape(len(frame), len(styles)),
                       frame[[f"target_{style}_5" for style in styles]])
    assert np.allclose(weights, [1, 1, 1, .5, .5, .5])


def test_reused_style_xgb_selection_trains_on_all_rows_without_new_embargo(monkeypatch):
    styles = ["style_1", "style_2", "style_3"]
    dates = pd.date_range("2023-01-02", periods=30, freq="B")
    rng = np.random.default_rng(44)
    frame = pd.DataFrame({"decision_date": dates, "sample_weight": 1.0,
                          "benchmark_volatility_20": .02,
                          "benchmark_return_20": rng.normal(0, .01, len(dates))})
    for style in styles:
        frame[f"{style}_momentum_5"] = rng.normal(size=len(frame))
        frame[f"{style}_momentum_20"] = rng.normal(size=len(frame))
        frame[f"{style}_volatility_5"] = rng.uniform(0, .03, len(frame))
        frame[f"target_{style}_5"] = rng.normal(0, .01, len(frame))

    def fail_if_split(*_args, **_kwargs):
        raise AssertionError("季度重训不应再次建立隔离验证集")

    monkeypatch.setattr(DualHorizonXGBoost, "_purged_time_split", fail_if_split)
    model = DualHorizonXGBoost(styles, 5).fit(
        frame, selected_trees=2, validation_rank_ic=.1,
        selection_fit_end=dates[5], selection_validation_start=dates[26])
    assert model.best_iterations == [2]
    assert np.isclose(model.validation_rank_ic, .1)
    assert len(model.predict(frame)) == len(frame)


def test_negative_realized_xgb_icir_disables_both_horizons_without_future_data():
    date = pd.Timestamp("2024-04-01")
    exits = pd.date_range("2024-03-01", periods=10, freq="B")
    history = pd.concat([
        pd.DataFrame({"decision_date": exits - pd.offsets.BDay(horizon),
                      "label_exit_date": exits, "horizon": horizon,
                      "model_rank_ic": np.linspace(-.30, -.05, len(exits))})
        for horizon in (5, 20)], ignore_index=True)
    test = pd.DataFrame({"decision_date": [date], "benchmark_volatility_20": [.02],
                         "benchmark_return_20": [.01]})
    result = StyleRotationPipeline(None)._dynamic_horizon_weights(
        test, test, ["style_1", "style_2", "style_3"], .3, .6, history)
    assert result.loc[0, "model_ic_observations_5"] == 10
    assert result.loc[0, "model_ic_observations_20"] == 10
    assert result.loc[0, "max_ic_information_date_5"] < date
    assert result.loc[0, "max_ic_information_date_20"] < date
    assert result.loc[0, "risk_off"]
    assert np.isclose(result.loc[0, "weight_5"] + result.loc[0, "weight_20"], 1.0)


def test_rolling_pca_uses_two_prior_quarters_and_aligns_components():
    rng = np.random.default_rng(101)
    rows = []
    dates = pd.to_datetime(["2021-07-01", "2021-10-01", "2022-01-04", "2022-04-01"])
    for date in dates:
        for symbol in range(20):
            rows.append([date, f"{symbol:06d}", *rng.normal(size=6)])
    frame = pd.DataFrame(rows, columns=["factor_date", "symbol", *[f"f{i}" for i in range(6)]])
    pipeline = StyleRotationPipeline(None)
    pcas, metadata = pipeline._rolling_pcas(
        frame, [f"f{i}" for i in range(6)],
        [pd.Period("2022Q1", freq="Q"), pd.Period("2022Q2", freq="Q")])

    assert metadata[pd.Period("2022Q1", freq="Q")]["fit_quarters"] == ["2021Q3", "2021Q4"]
    assert pd.Timestamp(metadata[pd.Period("2022Q1", freq="Q")]["fit_end"]) < pd.Timestamp("2022-01-01")
    assert metadata[pd.Period("2022Q2", freq="Q")]["fit_quarters"] == ["2021Q4", "2022Q1"]
    assert len(metadata[pd.Period("2022Q2", freq="Q")]["alignment"]) == len(pcas[pd.Period("2022Q2", freq="Q")].style_names)
    assert np.isfinite(pcas[pd.Period("2022Q1", freq="Q")].loadings).all()


def test_rolling_pca_starts_after_first_complete_lookback_window():
    processed = pd.DataFrame({"factor_date": pd.to_datetime([
        "2020-01-02", "2020-04-01", "2020-07-01", "2020-10-09"])
    })

    quarters = StyleRotationPipeline._eligible_pca_quarters(
        processed, pd.Period("2020Q1", freq="Q"), pd.Period("2020Q4", freq="Q"))

    assert quarters == [pd.Period("2020Q3", freq="Q"), pd.Period("2020Q4", freq="Q")]
