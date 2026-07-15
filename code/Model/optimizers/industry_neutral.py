from imports import *
from optimizers.base import ScoreOptimizer


class IndustryNeutralOptimizer(ScoreOptimizer):
    """把票池内预测分数投影到行业哑变量的正交补空间。

    每日、每个行业执行 ``score - industry_mean(score)``，随后只乘一个全局
    正常数恢复当日原始分数波动。因此行业内优化后分数均值严格为0，同时保留
    行业内排序。该约束作用于选股分数，不改变BackTest的等权、换手和成交规则。
    """

    name = "industry_neutral"
    SYMBOL_COLUMNS = ("ticker", "symbol", "code")
    INDUSTRY_COLUMNS = ("industry", "industry_code", "sector")
    DATE_COLUMNS = ("date", "factor_date", "effective_date")

    def __init__(self, path: Path):
        self.path = Path(path)
        self.mapping = self.load_mapping(self.path)

    @classmethod
    def load_mapping(cls, path: Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"行业分类文件不存在：{path}")
        if path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        else:
            raise ValueError("行业分类仅支持csv或parquet")
        frame = frame.reset_index() if not isinstance(frame.index, pd.RangeIndex) else frame.copy()
        lower = {str(column).strip().lower(): column for column in frame.columns}
        symbol_column = next((lower[name] for name in cls.SYMBOL_COLUMNS if name in lower), None)
        industry_column = next((lower[name] for name in cls.INDUSTRY_COLUMNS if name in lower), None)
        date_column = next((lower[name] for name in cls.DATE_COLUMNS if name in lower), None)
        if symbol_column is None or industry_column is None:
            raise ValueError("行业文件必须包含ticker/symbol/code和industry/industry_code/sector列")
        columns = [symbol_column, industry_column] + ([date_column] if date_column else [])
        out = frame[columns].rename(columns={symbol_column: "symbol", industry_column: "industry"})
        if date_column:
            out = out.rename(columns={date_column: "date"})
            out["date"] = pd.to_datetime(out.date, errors="coerce")
            if out.date.isna().any():
                raise ValueError("行业文件date列包含无法解析的日期")
        out["symbol"] = out.symbol.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        out["industry"] = out.industry.astype("string").str.strip()
        if out.symbol.eq("").any() or out.industry.isna().any() or out.industry.eq("").any():
            raise ValueError("行业文件包含空ticker或空industry")
        keys = ["symbol", "date"] if "date" in out else ["symbol"]
        conflicting = out.groupby(keys, dropna=False).industry.nunique().gt(1)
        if conflicting.any():
            raise ValueError("行业文件同一股票同一生效日存在多个行业")
        return out.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)

    def _industry_matrix(self, dates: pd.DatetimeIndex, symbols: pd.Index) -> pd.DataFrame:
        symbols = pd.Index(symbols.astype(str).str.zfill(6))
        if "date" not in self.mapping:
            static = self.mapping.set_index("symbol").industry
            values = np.tile(static.reindex(symbols).to_numpy(), (len(dates), 1))
            return pd.DataFrame(values, index=dates, columns=symbols)
        wide = self.mapping.pivot(index="date", columns="symbol", values="industry").sort_index()
        # 只向后使用已经生效的行业分类，严禁bfill引入未来行业信息。
        aligned_index = wide.index.union(dates).sort_values()
        return wide.reindex(aligned_index).ffill().reindex(index=dates, columns=symbols)

    def transform(self, scores: pd.DataFrame, pool: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not isinstance(scores.index, pd.DatetimeIndex):
            raise ValueError("行业中性优化要求分数索引为DatetimeIndex")
        scores = scores.copy()
        scores.columns = scores.columns.astype(str).str.zfill(6)
        pool = pool.reindex(index=scores.index, columns=scores.columns).fillna(False).astype(bool)
        industries = self._industry_matrix(scores.index, scores.columns)
        optimized = scores.copy()
        audits = []
        for date in scores.index:
            eligible = pool.loc[date] & scores.loc[date].notna()
            industry = industries.loc[date]
            covered = eligible & industry.notna()
            missing = int(eligible.sum() - covered.sum())
            if missing:
                examples = list(scores.columns[eligible & industry.isna()][:5])
                raise ValueError(f"{date.date()} 行业覆盖缺失{missing}只，示例：{examples}")
            if not covered.any():
                audits.append({"date": date, "eligible": 0, "covered": 0,
                               "industry_count": 0, "max_abs_industry_mean": np.nan})
                continue
            raw = scores.loc[date, covered].astype(float)
            groups = industry[covered]
            neutral = raw - raw.groupby(groups).transform("mean")
            raw_std, neutral_std = raw.std(ddof=0), neutral.std(ddof=0)
            if np.isfinite(raw_std) and np.isfinite(neutral_std) and neutral_std > 1e-12:
                neutral *= raw_std / neutral_std
            optimized.loc[date, covered] = neutral
            industry_means = neutral.groupby(groups).mean()
            audits.append({"date": date, "eligible": int(eligible.sum()),
                           "covered": int(covered.sum()),
                           "industry_count": int(groups.nunique()),
                           "max_abs_industry_mean": float(industry_means.abs().max()),
                           "raw_score_std": float(raw_std),
                           "optimized_score_std": float(neutral.std(ddof=0))})
        return optimized, pd.DataFrame(audits)
