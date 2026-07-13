"""回测的全部可调参数。"""
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class BacktestConfig:
    data_dir: Path = ROOT / "BackTestData"
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

    @property
    def max_replacements(self) -> int:
        return int(self.holding_count * self.max_turnover_ratio)


MARKETS = {
    "A500": ("pool_A500.parquet", "Benchmark_A500.parquet"),
    "ZZ1000": ("pool_zz1000.parquet", "Benchmark_zz1000.parquet"),
    "ALL_A500": (None, "Benchmark_A500.parquet"),
    "ALL_ZZ1000": (None, "Benchmark_zz1000.parquet"),
}

