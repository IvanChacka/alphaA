"""Composable research services used by the web console and batch jobs."""

from .analysis import FactorResearchEngine, ICAnalyzer, LayeredBacktester
from .data import DataCatalog, DataSourceFactory, LocalParquetSource, TushareSource
from .factors import Factor, FactorPipeline, FactorRegistry, factor_registry
from .models import MODEL_CATALOG, framework_manifest
from .portfolio import PortfolioConfig, PortfolioEngine, performance_metrics

__all__ = [
    "DataCatalog", "DataSourceFactory", "LocalParquetSource", "TushareSource",
    "Factor", "FactorPipeline", "FactorRegistry", "factor_registry",
    "ICAnalyzer", "LayeredBacktester", "FactorResearchEngine",
    "PortfolioConfig", "PortfolioEngine", "performance_metrics",
    "MODEL_CATALOG", "framework_manifest",
]
