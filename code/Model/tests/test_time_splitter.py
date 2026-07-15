from imports import *
from rolling_ml.time_splitter import TimeSplitter

def test_four_quarters_cover_year_without_overlap():
    cal=pd.bdate_range("2023-01-01","2024-12-31");parts=TimeSplitter(2024).quarters(cal)
    assert len(parts)==4
    dates=[]
    for p in parts:dates.extend(cal[(cal>=p.prediction_start)&(cal<=p.prediction_end)])
    assert len(dates)==len(set(dates))==sum(cal.year==2024)

