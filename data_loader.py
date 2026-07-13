"""宽表行情数据加载与一致化。行是交易日，列是六位股票代码。"""
from pathlib import Path
import pandas as pd
from config import BacktestConfig, MARKETS


class DataError(RuntimeError):
    pass


class MarketData:
    def __init__(self, cfg: BacktestConfig):
        self.cfg = cfg

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

    def load(self) -> dict[str, pd.DataFrame | pd.Series]:
        d = self.cfg.data_dir
        pool_name, benchmark_name = MARKETS[self.cfg.market]
        names = {
            "twap": "twap.parquet", "factor": "factordata.parquet",
            "close": "closePrice.parquet", "adj": "accumAdjFactor.parquet",
            "st": "ST.parquet",
        }
        out = {k: self._read(d / v) for k, v in names.items()}
        out["pool"] = self._read(d / pool_name) if pool_name else out["factor"].notna()
        benchmark = self._read(d / benchmark_name).iloc[:, 0]
        out["benchmark"] = benchmark.rename("benchmark")
        common_dates = out["twap"].index
        for key in ("factor", "close", "adj", "st", "pool"):
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
        for key in ("twap", "factor", "close", "adj", "st", "pool"):
            out[key] = out[key].reindex(index=common_dates, columns=common_codes)
        out["benchmark"] = out["benchmark"].reindex(common_dates).ffill()
        return out
