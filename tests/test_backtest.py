import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from backtest import BacktestEngine
from config import BacktestConfig


class EngineRulesTest(unittest.TestCase):
    def test_lag_lots_st_suspension_and_turnover(self):
        dates = pd.date_range("2024-01-02", periods=4, freq="B")
        codes = ["000001", "000002", "300001", "688001"]
        base = pd.DataFrame(10.0, index=dates, columns=codes)
        factor = pd.DataFrame([[4,3,2,1], [1,2,4,3], [1,4,3,2], [1,2,3,4]], index=dates, columns=codes)
        twap = base.copy(); twap.loc[dates[1], "000002"] = np.nan
        st = pd.DataFrame(False, index=dates, columns=codes); st.loc[dates[1], "300001"] = True
        adj = pd.DataFrame(1.0, index=dates, columns=codes); adj.loc[dates[2]:, "000001"] = 2.0
        data = {"twap":twap, "factor":factor, "close":base, "adj":adj, "st":st,
                "pool":pd.DataFrame(True,index=dates,columns=codes), "benchmark":pd.Series(range(4),index=dates)+100}
        cfg = replace(BacktestConfig(), initial_cash=100_000, holding_count=2, max_turnover_ratio=.5)
        result = BacktestEngine(data, cfg).run()
        buys = result.orders[result.orders.side == "BUY"]
        self.assertIn("000001", set(buys.code))  # 使用首日因子
        self.assertTrue(all(r.shares % (200 if r.code.startswith("688") else 100) == 0 for _,r in buys.iterrows()))
        self.assertLessEqual(result.account.holding_count.max(), 2)


if __name__ == "__main__": unittest.main()
