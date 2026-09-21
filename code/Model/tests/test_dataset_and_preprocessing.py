from imports import *
from rolling_ml.dataset_builder import DatasetBuilder
from rolling_ml.data_loader import ExistingDataAdapter
from rolling_ml.preprocessing import FeaturePreprocessor
from style_rotation.preprocessing import CrossSectionalPreprocessor
import rolling_ml.data_loader as data_loader


def test_unlabeled_tail_is_kept_for_prediction():
    factors = pd.DataFrame({"factor_date": pd.to_datetime(["2024-12-30", "2024-12-31"]),
                            "symbol": ["000001", "000001"], "alpha": [1.0, 2.0]})
    labels = pd.DataFrame(columns=["factor_date", "symbol", "label", "label_entry_date", "label_exit_date"])
    bundle = DatasetBuilder().build(factors, labels)
    assert len(bundle.data) == 2
    assert bundle.data.label.isna().all()


def test_preprocessor_passes_clean_factors_through_unchanged():
    train = pd.DataFrame({"factor_date": pd.to_datetime(["2024-01-02"] * 3), "alpha": [1.0, 2.0, 3.0]})
    processor = FeaturePreprocessor(scale=False)
    transformed = processor.fit_transform(FeaturePreprocessor.cross_sectional(train, ["alpha"]))
    assert transformed.dtype == np.float32
    assert np.array_equal(transformed.ravel(), np.array([1.0, 2.0, 3.0], dtype=np.float32))


def test_linear_preprocessor_restores_standardization():
    values = pd.DataFrame({"alpha": [1.0, 2.0, 3.0]})
    transformed = FeaturePreprocessor(scale=True).fit_transform(values)
    assert np.isclose(transformed.mean(), 0.0, atol=1e-6)
    assert np.isclose(transformed.std(ddof=0), 1.0, atol=1e-6)


def test_preprocessor_standardizes_observed_values_before_filling_missing():
    train = pd.DataFrame({"alpha": [1.0, 3.0, np.nan]})
    test = pd.DataFrame({"alpha": [5.0, np.nan]})
    processor = FeaturePreprocessor(scale=True)
    transformed_train = processor.fit_transform(train)
    transformed_test = processor.transform(test)
    assert np.allclose(transformed_train.ravel(), [-1.0, 1.0, 0.0])
    # Uses training mean=2 and std=1; the test sample is not used to refit.
    assert np.allclose(transformed_test.ravel(), [3.0, 0.0])


def test_unscaled_preprocessor_fills_only_missing_values_at_model_boundary():
    values = pd.DataFrame({"alpha": [1.0, np.nan, -2.0]})
    transformed = FeaturePreprocessor(scale=False).fit_transform(values)
    assert np.array_equal(transformed.ravel(), np.array([1.0, 0.0, -2.0], dtype=np.float32))


def test_preprocessor_rejects_infinite_factors_instead_of_silently_filling():
    dirty = pd.DataFrame({"alpha": [1.0, np.inf]})
    with np.testing.assert_raises_regex(ValueError, "Inf=1"):
        FeaturePreprocessor().fit_transform(dirty)


def test_data_adapter_drops_all_missing_rows_but_preserves_partial_missing():
    frame = pd.DataFrame({"factor_date": pd.to_datetime(["2024-01-02"] * 3),
                          "symbol": ["000001", "000002", "000003"],
                          "alpha_0": [1.0, np.nan, np.nan],
                          "alpha_1": [np.nan, 2.0, np.nan]})
    result, stats = ExistingDataAdapter.prepare_missing_factors(frame, ["alpha_0", "alpha_1"])
    assert result.symbol.tolist() == ["000001", "000002"]
    assert result[["alpha_0", "alpha_1"]].isna().sum().sum() == 2
    assert not any(column.endswith("_missing") for column in result.columns)
    assert stats["all_factor_missing_rows_dropped"] == 1
    assert stats["partial_factor_missing_cells_preserved"] == 2


def test_monthly_factor_schema_dates_and_target_are_supported():
    dates = ExistingDataAdapter._dates(pd.Series([pd.Timestamp("2024-01-31")]))
    frame = pd.DataFrame({"factor_date": dates, "symbol": ["000001"],
                          "calc_size": [1.0], "r_shift": [0.05]})

    assert dates[0] == pd.Timestamp("2024-01-31")
    assert ExistingDataAdapter.feature_columns(frame) == ["calc_size"]


def test_data_adapter_filters_to_explicit_selected_features(tmp_path, monkeypatch):
    factor_path = tmp_path / "factors.parquet"
    market_path = tmp_path / "market.parquet"
    placeholder = tmp_path / "placeholder.parquet"
    calendar = tmp_path / "calendar.csv"
    pd.DataFrame({"date": [20240102, 20240102], "ticker": ["000001", "000002"],
                  "alpha_a": [1., 2.], "alpha_b": [3., 4.]}).to_parquet(factor_path)
    pd.DataFrame({"TradingDate": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                  "Stkcd": ["000001", "000001"], "price": [10., 11.]}).to_parquet(market_path)
    pd.DataFrame({"unused": [1]}).to_parquet(placeholder)
    calendar.write_text("calendarDate,isOpen\n2024-01-02,1\n", encoding="utf-8")
    monkeypatch.setattr(data_loader, "ADJ_CLOSE_MARKET_PATH", market_path)

    bundle = ExistingDataAdapter(
        factor_path, placeholder, placeholder, calendar,
        selected_features=["alpha_b"]).load("2024-01-02", "2024-01-02", all_market=True)

    assert ExistingDataAdapter.feature_columns(bundle.factors) == ["alpha_b"]
    assert "alpha_a" not in bundle.factors


def test_style_preprocessor_standardizes_daily_observations_before_filling():
    frame = pd.DataFrame({
        "factor_date": pd.to_datetime(["2024-01-02"] * 3),
        "symbol": ["000001", "000002", "000003"],
        "alpha": [1.0, 3.0, np.nan],
    })
    transformed = CrossSectionalPreprocessor().transform(frame, ["alpha"])
    assert np.allclose(transformed["alpha"].to_numpy(), [-1.0, 1.0, 0.0])
