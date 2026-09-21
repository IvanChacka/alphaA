"""Factor registry and cross-sectional processing pipeline."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd


class Factor(ABC):
    name = "factor"
    description = ""
    required_columns: tuple[str, ...] = ()

    @abstractmethod
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """Return a Series aligned to ``data.index``."""


class FactorRegistry:
    def __init__(self):
        self._items: dict[str, type[Factor]] = {}

    def register(self, factor: type[Factor] | None = None, *, name: str | None = None):
        def decorator(cls: type[Factor]):
            key = name or getattr(cls, "name", cls.__name__)
            if not key or key in self._items:
                raise ValueError(f"因子名称为空或已经注册：{key}")
            if not issubclass(cls, Factor):
                raise TypeError("注册对象必须继承 Factor 并实现 compute()")
            self._items[key] = cls
            return cls
        return decorator(factor) if factor is not None else decorator

    def create(self, name: str, **parameters) -> Factor:
        if name not in self._items:
            raise KeyError(f"未注册因子：{name}")
        return self._items[name](**parameters)

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": name, "description": cls.description,
                 "required_columns": list(cls.required_columns)}
                for name, cls in self._items.items()]


factor_registry = FactorRegistry()


@factor_registry.register
class ColumnFactor(Factor):
    """Expose one numeric input field through the standard factor pipeline."""
    name = "column"
    description = "直接使用数据源中的数值列，随后执行逐日截面 Z-Score。"

    def __init__(self, column: str):
        self.column = column
        self.required_columns = (column,)

    def compute(self, data: pd.DataFrame) -> pd.Series:
        return pd.to_numeric(data[self.column], errors="coerce")


@dataclass
class FactorRunResult:
    values: pd.DataFrame
    warnings: list[str]
    skipped: list[str]
    coverage: list[dict[str, Any]]


class FactorPipeline:
    def __init__(self, registry: FactorRegistry = factor_registry, winsor_limit: float = 5.0):
        self.registry = registry
        self.winsor_limit = float(winsor_limit)

    @staticmethod
    def zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
        def scale(group: pd.Series) -> pd.Series:
            valid = group.replace([np.inf, -np.inf], np.nan)
            std = valid.std(ddof=0)
            if not np.isfinite(std) or std <= 1e-12:
                return pd.Series(np.nan, index=group.index)
            return (valid - valid.mean()) / std
        return values.groupby(dates, group_keys=False).apply(scale)

    def run(self, data: pd.DataFrame, specifications: list[dict[str, Any]],
            date_column: str = "date", symbol_column: str = "symbol") -> FactorRunResult:
        required = {date_column, symbol_column}
        if not required.issubset(data.columns):
            raise ValueError(f"因子输入缺少字段：{sorted(required - set(data.columns))}")
        base = data[[date_column, symbol_column]].copy()
        base[date_column] = pd.to_datetime(base[date_column], errors="coerce")
        messages: list[str] = []
        skipped: list[str] = []
        coverage: list[dict[str, Any]] = []
        for spec in specifications:
            name = str(spec.get("name", ""))
            output = str(spec.get("output", name))
            params = dict(spec.get("parameters", {}))
            try:
                factor = self.registry.create(name, **params)
                missing = set(factor.required_columns) - set(data.columns)
                if missing:
                    raise KeyError(f"缺少原始字段 {sorted(missing)}")
                raw = factor.compute(data).replace([np.inf, -np.inf], np.nan)
                standardized = self.zscore(raw, base[date_column]).clip(
                    -self.winsor_limit, self.winsor_limit)
                valid = int(standardized.notna().sum())
                if valid == 0:
                    raise ValueError("没有有效截面观测")
                base[output] = standardized
                coverage.append({"factor": output, "valid": valid,
                                 "missing_rate": float(1 - valid / max(len(base), 1))})
            except Exception as exc:
                message = f"因子 {output} 已跳过：{exc}"
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                messages.append(message)
                skipped.append(output)
        return FactorRunResult(base, messages, skipped, coverage)
