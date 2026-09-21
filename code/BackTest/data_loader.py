"""宽表行情数据加载与一致化。行是交易日，列是六位股票代码。"""
from pathlib import Path
import numpy as np
import pandas as pd
from config import BacktestConfig, MARKETS, PROJECT_ROOT, available_signals


class DataError(RuntimeError):
    pass


class MarketData:
    def __init__(self, cfg: BacktestConfig, factor_override: pd.DataFrame | None = None):
        self.cfg = cfg
        self.factor_override = factor_override

    @staticmethod
    def _align_signal_dates(factor: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
        """Place calendar-period-end signals on the latest tradable date."""
        if factor.empty or calendar.empty:
            return factor
        positions = calendar.searchsorted(pd.DatetimeIndex(factor.index), side="right") - 1
        valid = positions >= 0
        aligned = factor.loc[valid].copy()
        aligned.index = calendar[positions[valid]]
        return aligned.groupby(level=0).last().sort_index()

    @staticmethod
    def _signal_eligibility(factor: pd.DataFrame) -> pd.DataFrame:
        """Carry the latest signal universe across non-signal trading days."""
        eligibility = factor.notna().astype("boolean")
        no_signal = factor.isna().all(axis=1)
        eligibility.loc[no_signal, :] = pd.NA
        return eligibility.ffill().fillna(False).astype(bool)

    @staticmethod
    def _price_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
        """Keep dates with a new market close; drop holiday carry-forward rows."""
        if prices.empty:
            return pd.DatetimeIndex([])
        previous = prices.shift()
        same = prices.eq(previous) | (prices.isna() & previous.isna())
        changed = ~same.all(axis=1)
        changed.iloc[0] = True
        return pd.DatetimeIndex(prices.index[changed.fillna(False)])

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise DataError(f"数据文件不存在：{path}")
        # Parquet 文件首尾都必须是 PAR1。提前检查可将“复制未完成/文件被截断”
        # 与字段或回测逻辑错误区分开来。
        if path.stat().st_size < 8:
            raise DataError(f"{path.name} 文件为空或不完整，请重新复制原始文件")
        with path.open("rb") as stream:
            header = stream.read(4)
            stream.seek(-4, 2)
            footer = stream.read(4)
        if header != b"PAR1" or footer != b"PAR1":
            raise DataError(
                f"{path.name} 不是完整的 Parquet 文件（当前大小 "
                f"{path.stat().st_size:,} 字节，文件尾缺少 PAR1）。"
                "请从原始数据源重新完整复制，等待复制结束后再运行回测；"
                "不要把文件分片或下载缓存直接改名为 .parquet。"
            )
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            raise DataError(f"无法读取 {path.name}：{exc}") from exc
        if not isinstance(frame.index, pd.DatetimeIndex):
            frame.index = pd.to_datetime(frame.index.astype(str), format="%Y%m%d", errors="coerce")
        else:
            frame.index = pd.to_datetime(frame.index)
        frame = frame.loc[frame.index.notna()]
        frame.columns = frame.columns.astype(str).str.zfill(6)
        return frame.sort_index()

    @staticmethod
    def _read_factor(path: Path, signal: str) -> pd.DataFrame:
        """将 (ticker, date) MultiIndex 因子长表转换为日期×股票宽表。"""
        try:
            available = available_signals()
            if signal == "default_factor":
                factor_name = next(
                    key.split(":", 1)[1] for key in available if key.startswith("factor:")
                )
            elif signal.startswith("factor:"):
                factor_name = signal.split(":", 1)[1]
            else:
                raise DataError(f"不是因子信号：{signal}")

            # 只读取一个因子列，避免一次载入全部30个因子占用大量内存。
            frame = pd.read_parquet(path, columns=[factor_name])
        except (KeyError, StopIteration) as exc:
            raise DataError(f"因子字段不存在：{signal}") from exc
        except DataError:
            raise
        except Exception as exc:
            raise DataError(f"无法读取因子 {signal}：{exc}") from exc

        if not isinstance(frame.index, pd.MultiIndex):
            raise DataError("factordata.parquet 应使用 ticker、date 两级索引")
        if not {"ticker", "date"}.issubset(frame.index.names):
            raise DataError(
                f"因子索引应为 ticker、date，实际为：{frame.index.names}"
            )

        series = frame[factor_name]
        wide = series.unstack("ticker")
        wide.index = pd.to_datetime(
            wide.index.astype(str), format="%Y%m%d", errors="coerce"
        )
        wide = wide.loc[wide.index.notna()]
        wide.columns = wide.columns.astype(str).str.zfill(6)
        return wide.sort_index()

    @staticmethod
    def _read_uploaded_signal(path: Path) -> pd.DataFrame:
        """读取用户上传的长表信号，要求包含 date、ticker 和预测值列。"""
        try:
            raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_parquet(path)
            if isinstance(raw.index, pd.MultiIndex):
                raw = raw.reset_index()
            elif raw.index.name in {"date", "ticker"}:
                raw = raw.reset_index()
        except Exception as exc:
            raise DataError(f"无法读取上传信号 {path.name}：{exc}") from exc

        required = {"date", "ticker"}
        if not required.issubset(raw.columns):
            raise DataError(f"{path.name} 必须包含 date、ticker 字段")
        values = [c for c in raw.columns if c not in required]
        numeric = [c for c in values if pd.api.types.is_numeric_dtype(raw[c])]
        if not numeric:
            raise DataError(f"{path.name} 缺少数值型因子/预测字段")
        value = numeric[-1]
        raw["ticker"] = raw["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        raw["date"] = pd.to_datetime(
            raw["date"].astype(str).str.replace(r"\.0$", "", regex=True),
            format="%Y%m%d", errors="coerce",
        )
        raw = raw.dropna(subset=["date", "ticker", value])
        return raw.pivot_table(index="date", columns="ticker", values=value, aggfunc="last").sort_index()

    def load(self) -> dict[str, pd.DataFrame | pd.Series]:
        d = self.cfg.data_dir
        pool_name, benchmark_name = MARKETS[self.cfg.market]
        if self.cfg.market == "ALL_MARKET":
            return self._load_all_market(d)
        names = {
            "twap": "twap.parquet",
            "close": "closePrice.parquet", "adj": "accumAdjFactor.parquet",
            "st": "ST.parquet",
        }
        out = {k: self._read(d / v) for k, v in names.items()}
        if self.factor_override is not None:
            out["factor"] = self.factor_override.copy()
            out["factor"].index = pd.to_datetime(out["factor"].index)
            out["factor"].columns = out["factor"].columns.astype(str).str.zfill(6)
        elif self.cfg.signal not in available_signals():
            raise DataError(f"未知因子/模型：{self.cfg.signal}")
        elif self.cfg.signal == "default_factor" or self.cfg.signal.startswith("factor:"):
            out["factor"] = self._read_factor(d / "factordata.parquet", self.cfg.signal)
        elif self.cfg.signal.startswith("upload:"):
            filename = self.cfg.signal.split(":", 1)[1]
            if Path(filename).name != filename:
                raise DataError("上传信号文件名不安全")
            out["factor"] = self._read_uploaded_signal(
                d / "uploaded_signals" / filename
            )
        else:
            model_dir = PROJECT_ROOT / "signal" / self.cfg.signal
            files = sorted(model_dir.glob("*.csv"))
            if not files:
                raise DataError(f"模型目录没有预测 CSV：{model_dir}")
            raw = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
            required = {"date", "ticker"}
            if not required.issubset(raw.columns):
                raise DataError("模型 CSV 必须包含 date 和 ticker 字段")
            values = [c for c in raw.columns if c not in required]
            if not values:
                raise DataError("模型 CSV 缺少预测值字段")
            raw["ticker"] = raw["ticker"].astype(str).str.zfill(6)
            raw["date"] = pd.to_datetime(raw["date"].astype(str), errors="coerce")
            out["factor"] = raw.pivot_table(index="date", columns="ticker", values=values[-1], aggfunc="last")
        out["pool"] = self._read(d / pool_name) if pool_name else out["factor"].notna()
        benchmark = self._read(d / benchmark_name).iloc[:, 0]
        out["benchmark"] = benchmark.rename("benchmark")
        common_dates = out["twap"].index
        for key in ("close", "adj", "st", "pool"):
            common_dates = common_dates.intersection(out[key].index)
        if self.cfg.start_date:
            common_dates = common_dates[common_dates >= pd.Timestamp(self.cfg.start_date)]
        if self.cfg.end_date:
            common_dates = common_dates[common_dates <= pd.Timestamp(self.cfg.end_date)]
        if len(common_dates) < 2:
            raise DataError("共同可用交易日少于 2 天")
        common_codes = out["twap"].columns
        for key in ("factor", "close", "adj", "st", "pool"):
            common_codes = common_codes.intersection(out[key].columns)
        if len(common_codes) < self.cfg.holding_count:
            raise DataError(f"共同股票数量 {len(common_codes)} 少于目标持仓 {self.cfg.holding_count}")
        for key in ("twap", "close", "adj", "st", "pool"):
            out[key] = out[key].reindex(index=common_dates, columns=common_codes)
        out["factor"] = self._align_signal_dates(out["factor"], common_dates).reindex(
            index=common_dates, columns=common_codes)
        out["benchmark"] = out["benchmark"].reindex(common_dates).ffill()
        return out

    def _load_all_market(self, data_dir: Path) -> dict[str, pd.DataFrame | pd.Series]:
        path = data_dir / "adj_close.pq" / "adj_close.pq" / "adj_close.parquet"
        filters = []
        if self.cfg.start_date:
            # Keep a short lead-in so a weekend/holiday month-end signal can be
            # attached to the preceding tradable date and executed on t+1.
            filters.append(("TradingDate", ">=", pd.Timestamp(self.cfg.start_date) - pd.Timedelta(days=10)))
        if self.cfg.end_date:
            filters.append(("TradingDate", "<=", pd.Timestamp(self.cfg.end_date)))
        raw = pd.read_parquet(path, filters=filters or None)
        required = {"TradingDate", "Stkcd", "price"}
        if not required.issubset(raw.columns):
            raise DataError(f"全市场复权收盘文件缺少字段：{required - set(raw.columns)}")
        raw["TradingDate"] = pd.to_datetime(raw["TradingDate"])
        raw["Stkcd"] = raw["Stkcd"].astype(str).str.replace(
            r"\.0$", "", regex=True).str.zfill(6)
        prices = raw.pivot_table(index="TradingDate", columns="Stkcd", values="price",
                                 aggfunc="last").sort_index()
        if prices.empty:
            raise DataError("所选区间没有全市场复权收盘数据")
        trading_dates = self._price_dates(prices)
        prices = prices.reindex(trading_dates)
        factor = (self.factor_override.copy() if self.factor_override is not None
                  else self._read_factor(data_dir / "factordata.parquet", self.cfg.signal))
        factor.index = pd.to_datetime(factor.index)
        factor.columns = factor.columns.astype(str).str.zfill(6)
        dates = prices.index
        if self.cfg.start_date and not factor.empty:
            first_signal = max(pd.Timestamp(self.cfg.start_date), pd.Timestamp(factor.index.min()))
            start_position = dates.searchsorted(first_signal, side="right") - 1
            if start_position >= 0:
                dates = dates[start_position:]
        codes = prices.columns.intersection(factor.columns)
        if len(dates) < 2 or len(codes) < self.cfg.holding_count:
            raise DataError("全市场价格与模型预测的共同日期或股票数量不足")
        prices = prices.reindex(index=dates, columns=codes)
        factor = self._align_signal_dates(factor, dates).reindex(index=dates, columns=codes)
        # The selected source is already the executable close-price universe.
        # Do not introduce a second, unrelated ST or index-membership universe.
        st = pd.DataFrame(False, index=dates, columns=codes)
        pool = prices.notna()
        # Compare the strategy with the same investable universe it can score.
        # The full price file contains names absent from the factor file; putting
        # those names in the benchmark would create an artificial shortfall for
        # every model, including PCA-XGB.
        eligibility = self._signal_eligibility(factor)
        benchmark_returns = prices.pct_change(fill_method=None).replace(
            [np.inf, -np.inf], np.nan).where(
                (eligibility & pool).shift(1, fill_value=False))
        benchmark = benchmark_returns.mean(axis=1, skipna=True).fillna(0).add(1).cumprod().mul(100)
        return {"twap": prices, "close": prices.copy(),
                "adj": pd.DataFrame(1.0, index=dates, columns=codes),
                "st": st,
                "factor": factor, "pool": pool,
                "benchmark": benchmark.rename("benchmark")}
