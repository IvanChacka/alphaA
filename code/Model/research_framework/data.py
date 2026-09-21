"""Data layer with automatic local Parquet and Tushare source detection."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


DATE_ALIASES = ("date", "factor_date", "TradingDate", "trade_date")
SYMBOL_ALIASES = ("symbol", "ticker", "Stkcd", "ts_code")


def _first(columns: set[str], aliases: tuple[str, ...]) -> str | None:
    return next((name for name in aliases if name in columns), None)


@dataclass(frozen=True)
class SourceProfile:
    provider: str
    name: str
    start: str | None
    end: str | None
    rows: int | None
    columns: list[str]
    date_column: str | None
    symbol_column: str | None
    available: bool = True
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class DataSource(ABC):
    provider = "unknown"

    @abstractmethod
    def inspect(self) -> SourceProfile:
        raise NotImplementedError

    @abstractmethod
    def load(self, start: str | None = None, end: str | None = None,
             columns: list[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError


class LocalParquetSource(DataSource):
    provider = "parquet"

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Parquet 数据源不存在：{self.path}")

    def _schema(self) -> tuple[list[str], str | None, str | None]:
        names = pq.ParquetFile(self.path).schema_arrow.names
        columns = set(names)
        return names, _first(columns, DATE_ALIASES), _first(columns, SYMBOL_ALIASES)

    def _filter_value(self, date_column: str, value: str):
        stamp = pd.Timestamp(value)
        field_type = pq.ParquetFile(self.path).schema_arrow.field(date_column).type
        return int(stamp.strftime("%Y%m%d")) if pa.types.is_integer(field_type) else stamp

    def load(self, start: str | None = None, end: str | None = None,
             columns: list[str] | None = None) -> pd.DataFrame:
        names, date_column, symbol_column = self._schema()
        selected = None if columns is None else list(dict.fromkeys(
            [*(name for name in (date_column,) if name), *columns]))
        unknown = set(selected or []) - set(names)
        if unknown:
            raise ValueError(f"数据源缺少字段：{sorted(unknown)}")
        filters = []
        if date_column and start:
            filters.append((date_column, ">=", self._filter_value(date_column, start)))
        if date_column and end:
            filters.append((date_column, "<=", self._filter_value(date_column, end)))
        frame = pd.read_parquet(self.path, columns=selected, filters=filters or None)
        # Some legacy factor files store date/ticker as a named pandas index;
        # Arrow still exposes them in the schema, so preserve that index here.
        if any(name and name in frame.index.names for name in (date_column, symbol_column)):
            frame = frame.reset_index()
        elif frame.index.name is not None or not isinstance(frame.index, pd.RangeIndex):
            frame = frame.reset_index()
        return frame.reset_index(drop=True)

    def inspect(self) -> SourceProfile:
        names, date_column, symbol_column = self._schema()
        parquet = pq.ParquetFile(self.path)
        metadata = parquet.metadata
        start = end = None
        if date_column:
            column_index = names.index(date_column)
            bounds = []
            for group_index in range(parquet.num_row_groups):
                statistics = metadata.row_group(group_index).column(column_index).statistics
                if statistics is not None and statistics.has_min_max:
                    bounds.extend((statistics.min, statistics.max))
            dates = pd.Series(bounds) if bounds else self.load(columns=[date_column])[date_column]
            parsed = pd.to_datetime(dates.astype(str).str.replace(r"\.0$", "", regex=True),
                                    format="%Y%m%d", errors="coerce")
            if parsed.isna().all():
                parsed = pd.to_datetime(dates, errors="coerce")
            if parsed.notna().any():
                start, end = str(parsed.min().date()), str(parsed.max().date())
        return SourceProfile(self.provider, self.path.name, start, end,
                             metadata.num_rows, names, date_column, symbol_column)


class TushareSource(DataSource):
    provider = "tushare"

    def __init__(self, endpoint: str = "daily", token: str | None = None,
                 parameters: dict[str, Any] | None = None):
        self.endpoint = endpoint
        self.token = token or os.getenv("TUSHARE_TOKEN", "")
        self.parameters = dict(parameters or {})

    def _client(self):
        if not self.token:
            raise ValueError("Tushare 数据源需要 TUSHARE_TOKEN")
        try:
            import tushare as ts
        except ImportError as exc:
            raise ImportError("使用 Tushare 数据源前请安装 tushare") from exc
        return ts.pro_api(self.token)

    def load(self, start: str | None = None, end: str | None = None,
             columns: list[str] | None = None) -> pd.DataFrame:
        method = getattr(self._client(), self.endpoint, None)
        if method is None:
            raise ValueError(f"未知 Tushare 接口：{self.endpoint}")
        params = dict(self.parameters)
        if start:
            params["start_date"] = pd.Timestamp(start).strftime("%Y%m%d")
        if end:
            params["end_date"] = pd.Timestamp(end).strftime("%Y%m%d")
        if columns:
            params["fields"] = ",".join(columns)
        return method(**params)

    def inspect(self) -> SourceProfile:
        return SourceProfile(self.provider, self.endpoint, None, None, None, [],
                             "trade_date", "ts_code", bool(self.token),
                             "已配置 TUSHARE_TOKEN" if self.token else "未配置 TUSHARE_TOKEN")


class DataSourceFactory:
    """Recognize a local path or a ``tushare://endpoint`` URI."""

    @staticmethod
    def create(source: str | Path | dict[str, Any]) -> DataSource:
        if isinstance(source, dict):
            provider = str(source.get("provider", "auto")).lower()
            if provider == "tushare":
                return TushareSource(str(source.get("endpoint", "daily")),
                                     source.get("token"), source.get("parameters"))
            source = source.get("path", "")
        text = str(source)
        if text.lower().startswith("tushare://"):
            return TushareSource(text.split("://", 1)[1] or "daily")
        path = Path(text)
        if path.suffix.lower() in {".parquet", ".pq"} or path.is_file():
            return LocalParquetSource(path)
        raise ValueError(f"无法自动识别数据源：{source}")


class DataCatalog:
    def __init__(self):
        self._sources: dict[str, DataSource] = {}

    def register(self, name: str, source: str | Path | dict[str, Any] | DataSource):
        if not name:
            raise ValueError("数据源名称不能为空")
        self._sources[name] = source if isinstance(source, DataSource) else DataSourceFactory.create(source)
        return self

    def get(self, name: str) -> DataSource:
        if name not in self._sources:
            raise KeyError(f"未注册数据源：{name}")
        return self._sources[name]

    def profiles(self) -> list[dict[str, Any]]:
        rows = []
        for name, source in self._sources.items():
            try:
                profile = source.inspect().as_dict()
            except Exception as exc:
                profile = SourceProfile(source.provider, name, None, None, None, [], None, None,
                                        False, str(exc)).as_dict()
            rows.append({"id": name, **profile})
        return rows
