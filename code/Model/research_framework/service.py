"""Application service joining data, factor and research layers."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .analysis import FactorResearchEngine, ICAnalyzer
from .data import DataCatalog, LocalParquetSource
from .factors import FactorPipeline
from .models import framework_manifest
from .portfolio import PortfolioConfig, PortfolioEngine, performance_metrics


def factor_columns(path: str | Path) -> list[str]:
    names = pq.ParquetFile(path).schema_arrow.names
    excluded = {"date", "ticker", "factor_date", "symbol", "TradingDate", "Stkcd", "r_shift"}
    return [name for name in names if name not in excluded]


def factor_catalog(path: str | Path, source: str) -> list[dict[str, Any]]:
    """Expose stable ordinals without inventing formulas absent from source data."""
    return [
        {"index": index, "name": name, "expression": None,
         "description": "原文件未提供表达式", "source": source}
        for index, name in enumerate(factor_columns(path), start=1)
    ]


def source_catalog(factor_sources: dict[str, Path], market_path: Path,
                   uploaded_dir: Path) -> DataCatalog:
    catalog = DataCatalog()
    for name, path in factor_sources.items():
        if Path(path).is_file():
            catalog.register(name, path)
    if Path(market_path).is_file():
        catalog.register("all_market_close", market_path)
    catalog.register("tushare_daily", {"provider": "tushare", "endpoint": "daily"})
    if uploaded_dir.exists():
        for path in sorted(uploaded_dir.glob("*.parquet")):
            catalog.register(f"uploaded:{path.name}", path)
    return catalog


class ResearchService:
    def __init__(self, factor_sources: dict[str, Path], market_path: Path,
                 uploaded_dir: Path):
        self.factor_sources = factor_sources
        self.market_path = Path(market_path)
        self.uploaded_dir = Path(uploaded_dir)

    def metadata(self) -> dict[str, Any]:
        catalog = source_catalog(self.factor_sources, self.market_path, self.uploaded_dir)
        manifest = framework_manifest()
        manifest["data_sources"] = catalog.profiles()
        manifest["factor_registry"] = FactorPipeline().registry.describe()
        for source in manifest["data_sources"]:
            if source["id"] != "all_market_close" and source.get("available"):
                path = (self.factor_sources.get(source["id"]) or
                        (self.uploaded_dir / source["id"].split(":", 1)[-1]
                         if source["id"].startswith("uploaded:") else None))
                source["factor_catalog"] = factor_catalog(path, source["id"]) if path else []
                source["factors"] = [item["name"] for item in source["factor_catalog"]]
                source["frequency"] = ("monthly" if source["id"] == "monthly_calc"
                                        else "daily" if source["id"] == "daily_alpha"
                                        else "custom")
        return manifest

    def resolve_factor_path(self, source: str, factor_file: str = "") -> Path:
        if factor_file:
            candidate = (self.uploaded_dir / Path(factor_file).name).resolve()
            if self.uploaded_dir.resolve() not in candidate.parents or not candidate.is_file():
                raise ValueError("所选上传因子文件不存在")
            return candidate
        path = self.factor_sources.get(source)
        if path is None or not Path(path).is_file():
            raise ValueError("所选因子数据源不存在")
        return Path(path)

    @staticmethod
    def _normalize_factor(raw: pd.DataFrame) -> pd.DataFrame:
        raw = raw.rename(columns={"factor_date": "date", "TradingDate": "date",
                                  "symbol": "ticker", "Stkcd": "ticker"})
        raw["date"] = pd.to_datetime(raw.date.astype(str).str.replace(r"\.0$", "", regex=True),
                                     format="%Y%m%d", errors="coerce").fillna(
                                         pd.to_datetime(raw.date, errors="coerce"))
        raw["ticker"] = raw.ticker.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        return raw.dropna(subset=["date", "ticker"])

    def analyze_factor(self, request: dict[str, Any]) -> dict[str, Any]:
        source = str(request.get("factor_source", "daily_alpha"))
        path = self.resolve_factor_path(source, str(request.get("factor_file", "")))
        available = factor_columns(path)
        requested = request.get("factors") or available[:1]
        selected = [str(name) for name in requested if str(name) in available]
        if not selected:
            raise ValueError("请至少选择一个有效因子")
        start = str(request.get("start", "2024-01-01"))
        end = str(request.get("end", "2024-12-31"))
        factor_source = LocalParquetSource(path)
        profile = factor_source.inspect()
        raw = factor_source.load(start, end, columns=[profile.date_column, profile.symbol_column, *selected])
        raw = self._normalize_factor(raw)
        specs = [{"name": "column", "output": name, "parameters": {"column": name}}
                 for name in selected]
        processed = FactorPipeline().run(raw, specs, "date", "ticker")
        price_source = LocalParquetSource(self.market_path)
        prices_raw = price_source.load(str(pd.Timestamp(start) - pd.Timedelta(days=10)),
                                       str(pd.Timestamp(end) + pd.Timedelta(days=120)),
                                       columns=["Stkcd", "price"])
        prices_raw = prices_raw.rename(columns={"TradingDate": "date", "Stkcd": "ticker"})
        prices_raw["date"] = pd.to_datetime(prices_raw.date)
        prices_raw["ticker"] = prices_raw.ticker.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        prices = prices_raw.pivot_table(index="date", columns="ticker", values="price", aggfunc="last").sort_index()
        factor_frames = {name: processed.values.pivot_table(
            index="date", columns="ticker", values=name, aggfunc="last").sort_index()
            for name in selected if name in processed.values}
        results = {}
        holding = str(request.get("holding_period", "M")).upper()
        for name, frame in factor_frames.items():
            results[name] = FactorResearchEngine().run(frame, prices, holding)
            results[name]["portfolio"] = self._portfolio(frame, prices, request)
        correlations = processed.values[selected].corr(method="spearman") if factor_frames else pd.DataFrame()
        return {"source": source, "start": start, "end": end,
                "warnings": processed.warnings, "coverage": processed.coverage,
                "results": results,
                "correlation": {"labels": correlations.columns.tolist(),
                                "values": correlations.where(correlations.notna(), None).values.tolist()},
                "return_matrix": self._return_matrix(factor_frames, prices),
                "methodology": {"signal_time": "t 收盘后形成信号", "entry_time": "t+1",
                                "ic": "截面 Spearman Rank IC 与 Pearson IC",
                                "standardization": "逐日截面 Z-Score，极值截断到 +/-5"}}

    @staticmethod
    def _portfolio(factor: pd.DataFrame, prices: pd.DataFrame,
                   request: dict[str, Any]) -> dict[str, Any]:
        signal_dates = pd.DatetimeIndex(factor.index)
        entry_positions = prices.index.searchsorted(signal_dates, side="right")
        valid = entry_positions < len(prices)
        if not valid.any():
            raise ValueError("因子区间之后没有可执行交易日")
        mapped = factor.iloc[np.flatnonzero(valid)].copy()
        mapped.index = prices.index[entry_positions[valid]]
        mapped = mapped.groupby(level=0).last().sort_index()
        calendar = prices.index[(prices.index >= mapped.index.min()) &
                                (prices.index <= pd.Timestamp(request.get("end", prices.index.max())))]
        scores = mapped.reindex(calendar).ffill()
        one_day = prices.shift(-1).div(prices).sub(1).reindex(calendar)
        config = PortfolioConfig(
            rebalance_frequency=str(request.get("portfolio_frequency", "M")).upper(),
            top_n=int(request.get("top_n", 200)),
            weight_mode=str(request.get("weight_mode", "score")),
            commission_rate=float(request.get("commission_rate", .0003)),
            stamp_duty_rate=float(request.get("stamp_duty_rate", .0005)),
            max_turnover=float(request.get("max_turnover", 1.0)),
            max_drawdown=(float(request["max_drawdown"])
                          if request.get("max_drawdown") not in {None, ""} else None),
        )
        output = PortfolioEngine(config).run(scores, one_day)
        curve = output.pop("curve").reset_index()
        curve["gross_nav"] = curve.gross_return.add(1).cumprod()
        output["gross_metrics"] = performance_metrics(curve.gross_return)
        output["curve"] = [{**row, "date": str(pd.Timestamp(row["date"]).date())}
                           for row in curve.to_dict(orient="records")]
        output["config"] = config.__dict__
        return output

    @staticmethod
    def _return_matrix(factors: dict[str, pd.DataFrame], prices: pd.DataFrame) -> dict[str, Any]:
        if not factors:
            return {"x": [], "y": [], "values": []}
        names = list(factors)
        first = factors[names[0]]
        second = (factors[names[1]] if len(names) > 1 else
                  prices.pct_change(20, fill_method=None).reindex(first.index, method="ffill"))
        future = ICAnalyzer.returns_for_signal_dates(prices, first.index, 20)
        buckets = [[[] for _ in range(3)] for _ in range(3)]
        for date in first.index.intersection(second.index).intersection(future.index):
            sample = pd.concat([first.loc[date].rename("x"), second.loc[date].rename("y"),
                                future.loc[date].rename("r")], axis=1).dropna()
            if len(sample) < 30:
                continue
            sample["xb"] = pd.qcut(sample.x.rank(method="first"), 3, labels=False)
            sample["yb"] = pd.qcut(sample.y.rank(method="first"), 3, labels=False)
            for (xb, yb), block in sample.groupby(["xb", "yb"]):
                buckets[int(yb)][int(xb)].extend(block.r.tolist())
        values = [[float(np.mean(cell)) if cell else None for cell in row] for row in buckets]
        return {"x": ["低", "中", "高"], "y": ["低", "中", "高"], "values": values,
                "x_factor": names[0], "y_factor": names[1] if len(names) > 1 else "20日动量"}
