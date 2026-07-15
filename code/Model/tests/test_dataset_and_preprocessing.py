from imports import *
from rolling_ml.dataset_builder import DatasetBuilder
from rolling_ml.data_loader import ExistingDataAdapter
from rolling_ml.preprocessing import FeaturePreprocessor


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


def test_preprocessor_rejects_dirty_factors_instead_of_filling():
    dirty = pd.DataFrame({"factor_date": pd.to_datetime(["2024-01-02"] * 2), "alpha": [1.0, np.nan]})
    with np.testing.assert_raises_regex(ValueError, "NaN=1"):
        FeaturePreprocessor().fit_transform(FeaturePreprocessor.cross_sectional(dirty, ["alpha"]))


def test_missing_factors_are_zero_filled_once_before_models():
    frame = pd.DataFrame({"factor_date": pd.to_datetime(["2024-01-02"] * 3),
                          "symbol": ["000001", "000002", "000003"],
                          "alpha_0": [1.0, np.nan, np.nan],
                          "alpha_1": [np.nan, 2.0, np.nan]})
    result, stats = ExistingDataAdapter.prepare_missing_factors(frame, ["alpha_0", "alpha_1"])
    assert result.symbol.tolist() == ["000001", "000002"]
    assert result[["alpha_0", "alpha_1"]].isna().sum().sum() == 0
    assert result[["alpha_0", "alpha_1"]].to_numpy().tolist() == [[1.0, 0.0], [0.0, 2.0]]
    assert not any(column.endswith("_missing") for column in result.columns)
    assert stats["all_factor_missing_rows_dropped"] == 1
