"""回测的全部可调参数。"""
from dataclasses import dataclass
from pathlib import Path
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parents[1]


@dataclass(frozen=True)
class BacktestConfig:
    data_dir: Path = PROJECT_ROOT / "BackTestData"
    output_dir: Path = ROOT / "output"
    initial_cash: float = 100_000_000.0
    holding_count: int = 200
    max_turnover_ratio: float = 0.30
    fee_rate: float = 0.0014
    annual_days: int = 252
    risk_free_rate: float = 0.0
    start_date: str | None = None
    end_date: str | None = None
    market: str = "A500"
    signal: str = "default_factor"

    @property
    def max_replacements(self) -> int:
        return int(self.holding_count * self.max_turnover_ratio)


MARKETS = {
    "A500": ("pool_A500.parquet", "Benchmark_A500.parquet"),
    "ZZ1000": ("pool_zz1000.parquet", "Benchmark_zz1000.parquet"),
    "ALL_A500": (None, "Benchmark_A500.parquet"),
    "ALL_ZZ1000": (None, "Benchmark_zz1000.parquet"),
}


def available_signals() -> dict[str, str]:
    """返回网页和命令行可选择的因子/模型。

    新模型只需在项目 ``signal`` 目录下建立同名子目录并放入预测 CSV。
    CSV 至少包含 date、ticker，以及一个预测值字段（例如 mean）。
    """
    signals: dict[str, str] = {}
    factor_path = PROJECT_ROOT / "BackTestData" / "factordata.parquet"
    if factor_path.exists():
        try:
            names = pq.ParquetFile(factor_path).schema_arrow.names
            factors = [name for name in names if name not in {"ticker", "date"}]
            signals.update({f"factor:{name}": f"因子：{name}" for name in factors})
            if factors:
                # 保留旧命令兼容性，默认使用文件中的第一个因子。
                signals["default_factor"] = f"默认因子（{factors[0]}）"
        except Exception:
            signals["default_factor"] = "默认因子（factordata.parquet）"
    signal_root = PROJECT_ROOT / "signal"
    if signal_root.exists():
        for path in sorted(signal_root.iterdir()):
            if path.is_dir() and any(path.glob("*.csv")):
                signals[path.name] = f"模型：{path.name}"
    upload_root = PROJECT_ROOT / "BackTestData" / "uploaded_signals"
    if upload_root.exists():
        for path in sorted(upload_root.iterdir()):
            if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}:
                signals[f"upload:{path.name}"] = f"上传：{path.stem}"
    return signals
