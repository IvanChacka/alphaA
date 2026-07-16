from imports import *
from env import ADJ_FACTOR_PATH, CALENDAR_PATH, FACTOR_PATH, FEATURE_FORBIDDEN_PATTERNS, TWAP_PATH


@dataclass
class MLDataBundle:
    factors: pd.DataFrame
    real_twap: pd.DataFrame
    calendar: pd.DatetimeIndex
    quality: pd.DataFrame


class ExistingDataAdapter:
    """只读适配现有数据，不修改价格、票池或回测业务逻辑。"""

    def __init__(self, factor_path=FACTOR_PATH, twap_path=TWAP_PATH,
                 adj_path=ADJ_FACTOR_PATH, calendar_path=CALENDAR_PATH):
        self.factor_path, self.twap_path = Path(factor_path), Path(twap_path)
        self.adj_path, self.calendar_path = Path(adj_path), Path(calendar_path)

    @staticmethod
    def _dates(values) -> pd.DatetimeIndex:
        text = pd.Index(values).astype(str).str.replace(r"\.0$", "", regex=True)
        return pd.to_datetime(text, format="%Y%m%d", errors="coerce")

    @staticmethod
    def prepare_missing_factors(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, dict[str, int]]:
        """删除全因子缺失行，并将部分缺失保留到模型预处理阶段。

        缺失值不能在这里提前填0：线性模型先使用训练期非缺失观测拟合标准化
        参数，风格模型先计算当日截面统计量，之后才将剩余缺失值填为中性值0。
        """
        if not features:
            raise ValueError("因子文件没有数值因子列")
        any_available = np.zeros(len(frame), dtype=bool)
        original_missing = 0
        for feature in features:
            available = frame[feature].notna().to_numpy()
            any_available |= available
            original_missing += int((~available).sum())
        dropped = int((~any_available).sum())
        out = frame.loc[any_available].copy()
        partial_rows = np.zeros(len(out), dtype=bool)
        for feature in features:
            missing = out[feature].isna().to_numpy()
            partial_rows |= missing
        return out, {"original_factor_missing_cells": original_missing,
                     "all_factor_missing_rows_dropped": dropped,
                     "partial_factor_missing_rows_kept": int(partial_rows.sum()),
                     "partial_factor_missing_cells_preserved": int(out[features].isna().sum().sum())}

    def load(self, start: str | None = None, end: str | None = None) -> MLDataBundle:
        for path in (self.factor_path, self.twap_path, self.adj_path, self.calendar_path):
            if not path.exists():
                raise FileNotFoundError(f"缺少数据文件：{path}")
        filters = []
        if start: filters.append(("date", ">=", int(pd.Timestamp(start).strftime("%Y%m%d"))))
        if end: filters.append(("date", "<=", int(pd.Timestamp(end).strftime("%Y%m%d"))))
        factors = pd.read_parquet(self.factor_path, filters=filters or None).reset_index()
        if not {"ticker", "date"}.issubset(factors.columns):
            raise ValueError("因子数据必须包含 ticker、date 索引或字段")
        factors["symbol"] = factors.pop("ticker").astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        factors["factor_date"] = self._dates(factors.pop("date"))
        factors = factors.dropna(subset=["factor_date", "symbol"])
        if factors.duplicated(["factor_date", "symbol"]).any():
            raise ValueError("因子数据存在重复的日期和股票组合")
        original_features = self.feature_columns(factors)
        suspicious = [c for c in original_features if any(token in c.lower() for token in FEATURE_FORBIDDEN_PATTERNS)]
        if suspicious:
            raise ValueError(f"检测到疑似未来信息特征：{suspicious}")
        factors[original_features] = factors[original_features].astype("float32")
        factors, missing_stats = self.prepare_missing_factors(factors, original_features)
        features = self.feature_columns(factors)

        twap = pd.read_parquet(self.twap_path)
        adj = pd.read_parquet(self.adj_path)
        twap.index, adj.index = self._dates(twap.index), self._dates(adj.index)
        twap.columns = twap.columns.astype(str).str.zfill(6)
        adj.columns = adj.columns.astype(str).str.zfill(6)
        dates = twap.index.intersection(adj.index)
        symbols = twap.columns.intersection(adj.columns)
        twap, adj = twap.reindex(index=dates, columns=symbols), adj.reindex(index=dates, columns=symbols)
        real_twap = twap.div(adj.where(adj > 0)).astype("float32")

        raw_calendar = pd.read_csv(self.calendar_path)
        if not {"calendarDate", "isOpen"}.issubset(raw_calendar.columns):
            raise ValueError("calendar.csv 必须包含 calendarDate 和 isOpen")
        open_mask = raw_calendar["isOpen"].astype(str).str.lower().isin({"1", "true", "t", "yes"})
        calendar = pd.DatetimeIndex(pd.to_datetime(raw_calendar.loc[open_mask, "calendarDate"])).sort_values().unique()
        common = calendar.intersection(real_twap.index)
        real_twap = real_twap.reindex(common)
        if start:
            factors = factors[factors.factor_date >= pd.Timestamp(start)]
            real_twap = real_twap.loc[pd.Timestamp(start):]
        if end:
            factors = factors[factors.factor_date <= pd.Timestamp(end)]
            # 价格保留 end 之后两个交易日，用于测试期末 t1/t2 标签。

        factor_dates = pd.DatetimeIndex(factors.factor_date.unique())
        quality = pd.DataFrame([
            {"item": "factor_rows", "value": len(factors)},
            {"item": "original_feature_count", "value": len(original_features)},
            {"item": "feature_count", "value": len(features)},
            *({"item": key, "value": value} for key, value in missing_stats.items()),
            {"item": "factor_duplicate_count", "value": int(factors.duplicated(["factor_date", "symbol"]).sum())},
            {"item": "factor_twap_common_dates", "value": len(factor_dates.intersection(real_twap.index))},
            {"item": "twap_adj_common_symbols", "value": len(symbols)},
            {"item": "twap_dates", "value": len(real_twap)},
            {"item": "twap_missing_rate", "value": float(real_twap.isna().mean().mean())},
        ])
        return MLDataBundle(factors, real_twap, common, quality)

    @staticmethod
    def feature_columns(frame: pd.DataFrame) -> list[str]:
        excluded = {"factor_date", "symbol", "label", "label_entry_date", "label_exit_date"}
        return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]
