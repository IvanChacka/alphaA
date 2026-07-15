from imports import *
from rolling_ml.label_builder import LabelBuilder

def test_label_alignment():
    dates=pd.bdate_range("2024-01-02",periods=5);prices=pd.DataFrame({"000001":[10,11,12,13,14]},index=dates)
    labels=LabelBuilder().build(prices);row=labels[(labels.factor_date==dates[0])&(labels.symbol=="000001")].iloc[0]
    assert np.isclose(row.label,12/11-1);assert row.label_entry_date==dates[1];assert row.label_exit_date==dates[2]
    tail = labels[labels.factor_date.isin(dates[-2:])]
    assert tail.label.isna().all()
