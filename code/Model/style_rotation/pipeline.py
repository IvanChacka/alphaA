from imports import *
from env import (ACTIVE_POOLS, DATA_DIR, FACTOR_PATH, STYLE_MODELS, STYLE_PCA_START,
                 STYLE_PCA_WINDOW_QUARTERS, STYLE_XGB_MAX_ESTIMATORS,
                 STYLE_TIME_DECAY_HALFLIFE, STYLE_TIMING_TRAIN_START)
from rolling_ml.data_loader import ExistingDataAdapter
from rolling_ml.label_builder import LabelBuilder
from rolling_ml.metrics import daily_ic, layered_metrics
from style_rotation.models import (DualHorizonXGBoost, PCAStockXGBoost,
                                   PCAStockXGBoostTuner, StyleTimer)
from style_rotation.config import StyleRuntimeConfig
from style_rotation.pca_style import FixedPCAStyle
from style_rotation.preprocessing import CrossSectionalPreprocessor
from style_rotation.residual_model import ResidualRidge
from style_rotation.style_returns import StyleReturnBuilder
from style_rotation.timing import TimingDatasetBuilder


class StyleRotationPipeline:
    """端到端风格工程。输出仍是BackTestAdapter要求的t0股票连续分数。"""

    def __init__(self, logger, test_year: int = 2024, models: list[str] | None = None,
                 config: StyleRuntimeConfig | None = None, train_start: str | None = None,
                 train_end: str | None = None, test_start: str | None = None,
                 test_end: str | None = None, factor_path: str | Path | None = None,
                 pools: list[str] | tuple[str, ...] | None = None,
                 optuna_trials: int = 30,
                 selected_features: list[str] | tuple[str, ...] | None = None,
                 window_mode: str = "expanding", train_years: int = 4):
        self.logger, self.test_year = logger, test_year
        self.models = models or list(STYLE_MODELS)
        self.config = (config or StyleRuntimeConfig()).validate()
        self.train_start = pd.Timestamp(train_start or STYLE_TIMING_TRAIN_START)
        self.train_end = pd.Timestamp(train_end or f"{test_year - 1}-12-31")
        self.test_start = pd.Timestamp(test_start or f"{test_year}-01-01")
        self.test_end = pd.Timestamp(test_end or f"{test_year}-12-31")
        if not self.train_start <= self.train_end < self.test_start <= self.test_end:
            raise ValueError("训练区间必须早于测试区间，且日期顺序必须有效")
        self.factor_path = Path(factor_path or FACTOR_PATH)
        self.pools = list(pools or ACTIVE_POOLS)
        self.optuna_trials = int(optuna_trials)
        self.selected_features = tuple(dict.fromkeys(selected_features or ()))
        if window_mode not in {"expanding", "fixed"}:
            raise ValueError("window_mode必须是 expanding 或 fixed")
        if not 1 <= int(train_years) <= 20:
            raise ValueError("训练窗口年数必须在1到20之间")
        self.window_mode, self.train_years = window_mode, int(train_years)

    def _training_start(self, cutoff: pd.Timestamp) -> pd.Timestamp:
        if self.window_mode == "fixed":
            rolling_start = pd.Timestamp(cutoff) - pd.DateOffset(
                years=self.train_years) + pd.Timedelta(days=1)
            return max(self.train_start, rolling_start)
        return self.train_start

    @staticmethod
    def _pool_membership(pool: str) -> pd.DataFrame | None:
        """Load point-in-time membership; ALL_* intentionally uses the full factor universe."""
        filenames = {"A500": "pool_A500.parquet", "ZZ1000": "pool_zz1000.parquet"}
        if pool.startswith("ALL_"):
            return None
        if pool not in filenames:
            raise ValueError(f"不支持的股票池：{pool}")
        membership = pd.read_parquet(DATA_DIR / filenames[pool])
        membership.index = ExistingDataAdapter._dates(membership.index)
        membership = membership.loc[membership.index.notna()].sort_index()
        membership.columns = membership.columns.astype(str).str.replace(
            r"\.0$", "", regex=True).str.zfill(6)
        return membership.astype("boolean")

    @staticmethod
    def _inside_pool(frame: pd.DataFrame, membership: pd.DataFrame | None) -> pd.DataFrame:
        if membership is None or frame.empty:
            return frame.copy()
        dates = pd.DatetimeIndex(frame.factor_date)
        symbols = frame.symbol.astype(str).str.zfill(6)
        date_codes = membership.index.get_indexer(dates)
        symbol_codes = membership.columns.get_indexer(symbols)
        valid = (date_codes >= 0) & (symbol_codes >= 0)
        included = np.zeros(len(frame), dtype=bool)
        positions = np.flatnonzero(valid)
        if len(positions):
            values = membership.to_numpy(dtype=bool, na_value=False, copy=False)
            included[positions] = values[date_codes[positions], symbol_codes[positions]]
        return frame.loc[included].copy()

    @staticmethod
    def _pool_benchmark_path(pool: str) -> Path:
        name = "Benchmark_A500.parquet" if pool in {"A500", "ALL_A500"} else "Benchmark_zz1000.parquet"
        return DATA_DIR / name

    @staticmethod
    def _benchmark(path: Path) -> pd.Series | None:
        if not path.exists(): return None
        frame = pd.read_parquet(path)
        if isinstance(frame, pd.Series): series = frame
        elif frame.shape[1] == 1: series = frame.iloc[:, 0]
        else: series = frame.select_dtypes("number").iloc[:, 0]
        series.index = pd.to_datetime(series.index.astype(str).str.replace(r"\.0$", "", regex=True), errors="coerce")
        return series.sort_index()

    @staticmethod
    def _stock_scores(exposure: pd.DataFrame, weights: pd.DataFrame, styles: list[str]) -> pd.DataFrame:
        merged = exposure.merge(weights, left_on="factor_date", right_on="decision_date", suffixes=("_x", "_w"))
        score = sum(merged[f"{s}_x"] * merged[f"{s}_w"] for s in styles)
        out = merged[["factor_date", "symbol"]].copy()
        out["prediction"] = score
        out["prediction"] = out.groupby("factor_date").prediction.transform(
            lambda x: (x - x.mean()) / (x.std(ddof=0) if x.std(ddof=0) > 1e-12 else 1.0))
        return out

    @staticmethod
    def _cross_style_ic(row: pd.Series, styles: list[str], horizon: int) -> float:
        signal = np.asarray([row.get(f"{style}_momentum_{horizon}", np.nan) for style in styles], dtype=float)
        target = np.asarray([row.get(f"target_{style}_{horizon}", np.nan) for style in styles], dtype=float)
        valid = np.isfinite(signal) & np.isfinite(target)
        if valid.sum() < 3 or np.std(signal[valid]) <= 1e-12 or np.std(target[valid]) <= 1e-12:
            return np.nan
        return float(stats.spearmanr(signal[valid], target[valid]).statistic)

    @staticmethod
    def _mean_forecast_ic(forecast: pd.DataFrame, actual: pd.DataFrame,
                          styles: list[str], horizon: int) -> float:
        """按日计算预测收益与同期限实现收益的跨风格RankIC。"""
        targets = actual[["decision_date", *[f"target_{style}_{horizon}" for style in styles]]]
        evaluated = forecast.merge(targets, on="decision_date", how="inner", validate="one_to_one")
        values = []
        for row in evaluated.itertuples(index=False):
            predicted = np.asarray([getattr(row, style) for style in styles], dtype=float)
            realized = np.asarray([getattr(row, f"target_{style}_{horizon}") for style in styles], dtype=float)
            valid = np.isfinite(predicted) & np.isfinite(realized)
            if valid.sum() >= 3 and np.std(predicted[valid]) > 1e-12 and np.std(realized[valid]) > 1e-12:
                values.append(stats.spearmanr(predicted[valid], realized[valid]).statistic)
        return float(np.nanmean(values)) if values else np.nan

    @staticmethod
    def _daily_forecast_ic(forecast: pd.DataFrame, actual: pd.DataFrame,
                           styles: list[str], horizon: int) -> pd.DataFrame:
        """生成逐日XGB跨风格IC，并记录该IC真正可获得的标签退出日。"""
        target_columns = [f"target_{style}_{horizon}" for style in styles]
        evaluated = forecast.merge(
            actual[["decision_date", f"label_exit_date_{horizon}", *target_columns]],
            on="decision_date", how="inner", validate="one_to_one")
        rows = []
        for row in evaluated.itertuples(index=False):
            predicted = np.asarray([getattr(row, style) for style in styles], dtype=float)
            realized = np.asarray([getattr(row, column) for column in target_columns], dtype=float)
            valid = np.isfinite(predicted) & np.isfinite(realized)
            rank_ic = np.nan
            if (valid.sum() >= 3 and np.std(predicted[valid]) > 1e-12 and
                    np.std(realized[valid]) > 1e-12):
                rank_ic = float(stats.spearmanr(predicted[valid], realized[valid]).statistic)
            rows.append({"decision_date": row.decision_date,
                         "label_exit_date": getattr(row, f"label_exit_date_{horizon}"),
                         "horizon": horizon, "model_rank_ic": rank_ic})
        return pd.DataFrame(rows)

    def _dynamic_horizon_weights(self, timing: pd.DataFrame, test: pd.DataFrame,
                                 styles: list[str], model_validation_ic_5: float = 0.0,
                                 model_validation_ic_20: float = 0.0,
                                 forecast_ic_history: pd.DataFrame | None = None) -> pd.DataFrame:
        """用已实现的XGB预测ICIR门控5日/20日权重，并用市场状态微调。

        固定验证IC只在尚无足够样本外预测记录时作为先验。达到最少观察数后，
        某期限滚动ICIR不为正则实时权重归零。两个期限都失效时只输出risk_off，
        预测分数回退到冻结的验证权重；是否暂停交易由最大回撤优化器决定。
        """
        forecast_ic_history = (pd.DataFrame(columns=["decision_date", "label_exit_date",
                                                      "horizon", "model_rank_ic"])
                               if forecast_ic_history is None
                               else forecast_ic_history.sort_values("decision_date"))
        rows = []
        for row in test.itertuples(index=False):
            date = row.decision_date
            def reliability(horizon: int, validation_prior: float):
                history = forecast_ic_history[
                    (forecast_ic_history.horizon == horizon) &
                    (forecast_ic_history.label_exit_date < date) &
                    forecast_ic_history.model_rank_ic.notna()].tail(self.config.icir_window)
                values = history.model_rank_ic
                minimum = min(10, self.config.icir_window)
                if len(values) >= minimum:
                    std = values.std(ddof=1)
                    value = float(values.mean() / std) if std > 1e-12 else 0.0
                    strength = max(value, 0.0)
                else:
                    value = np.nan
                    strength = max(float(validation_prior or 0.0), 0.0)
                information_date = history.label_exit_date.max() if len(history) else pd.NaT
                return strength, value, len(history), information_date
            strength5, icir5, count5, information5 = reliability(5, model_validation_ic_5)
            strength20, icir20, count20, information20 = reliability(20, model_validation_ic_20)
            past_vol = timing.loc[timing.decision_date <= date, "benchmark_volatility_20"].dropna().tail(252)
            current_vol = getattr(row, "benchmark_volatility_20", np.nan)
            vol_percentile = (float((past_vol <= current_vol).mean()) if len(past_vol) and np.isfinite(current_vol) else .5)
            market_return = abs(float(getattr(row, "benchmark_return_20", 0.0) or 0.0))
            trend_strength = market_return / max(float(current_vol or 0.0) * np.sqrt(20), 1e-6)
            short_raw = strength5 * (.5 + vol_percentile)
            medium_raw = strength20 * (.5 + min(trend_strength, 1.5))
            total = short_raw + medium_raw
            risk_off = total <= 1e-12
            if risk_off:
                short_raw = max(float(model_validation_ic_5 or 0.0), 0.0) * (.5 + vol_percentile)
                medium_raw = max(float(model_validation_ic_20 or 0.0), 0.0) * (
                    .5 + min(trend_strength, 1.5))
                total = short_raw + medium_raw
                if total <= 1e-12:
                    short_raw = medium_raw = total = 1.0
            weight5 = short_raw / total
            weight20 = medium_raw / total
            rows.append({"decision_date": date, "weight_5": weight5,
                         "weight_20": weight20, "rolling_icir_5": icir5,
                         "rolling_icir_20": icir20, "market_volatility_percentile": vol_percentile,
                         "market_trend_strength": trend_strength,
                         "risk_off": risk_off,
                         "model_ic_observations_5": count5,
                         "model_ic_observations_20": count20,
                         "model_validation_ic_5": model_validation_ic_5,
                         "model_validation_ic_20": model_validation_ic_20,
                         "max_ic_information_date_5": information5,
                         "max_ic_information_date_20": information20})
        return pd.DataFrame(rows)

    @staticmethod
    def _consensus_stock_scores(exposure: pd.DataFrame, forecast: pd.DataFrame,
                                styles: list[str], buy_confirmations: int = 1,
                                vote_quantile: float = .90
                                ) -> tuple[pd.DataFrame, pd.DataFrame]:
        merged = exposure.merge(forecast, left_on="factor_date", right_on="decision_date",
                                suffixes=("_exposure", "_forecast"))
        contributions = np.column_stack([
            merged[f"{style}_exposure"].to_numpy() * merged[f"{style}_forecast"].to_numpy()
            for style in styles])
        # 仅把每个风格截面上最明确的正/负贡献视为投票，微小贡献属于中性。
        bullish = np.zeros_like(contributions, dtype=bool)
        bearish = np.zeros_like(contributions, dtype=bool)
        for indices in merged.groupby("factor_date", sort=False).indices.values():
            block = contributions[indices]
            upper = np.nanquantile(block, vote_quantile, axis=0)
            lower = np.nanquantile(block, 1.0 - vote_quantile, axis=0)
            bullish[indices] = (block >= upper) & (block > 0)
            bearish[indices] = (block <= lower) & (block < 0)
        buy_votes = bullish.sum(axis=1)
        sell_votes = bearish.sum(axis=1)
        # 任意一个风格明确看多即可进入候选，不再要求看多票数压过看空票数。
        eligible = buy_votes >= buy_confirmations
        out = merged[["factor_date", "symbol"]].copy()
        bullish_score = np.where(bullish, contributions, 0.0).sum(axis=1)
        out["prediction"] = np.where(eligible, bullish_score + .05 * buy_votes, np.nan)
        out["prediction"] = out.groupby("factor_date").prediction.transform(
            lambda x: (x - x.mean()) / x.std(ddof=0) if x.notna().sum() > 1 and x.std(ddof=0) > 1e-12 else x)
        votes = out[["factor_date", "symbol"]].copy()
        votes["buy_votes"], votes["sell_votes"], votes["buy_eligible"] = buy_votes, sell_votes, eligible
        return out, votes

    @staticmethod
    def _standardize_score(frame: pd.DataFrame, column: str) -> pd.Series:
        return frame.groupby("factor_date")[column].transform(
            lambda values: ((values - values.mean()) / values.std(ddof=0)
                            if values.notna().sum() > 1 and values.std(ddof=0) > 1e-12
                            else values * 0.0))

    def _residual_score(self, model: ResidualRidge, frame: pd.DataFrame,
                        features: list[str]) -> pd.DataFrame:
        out = frame[["factor_date", "symbol"]].copy()
        out["residual_prediction_score"] = model.predict(frame, features)
        out["residual_prediction_score"] = self._standardize_score(out, "residual_prediction_score")
        return out

    @staticmethod
    def _combine_style_residual(style_score: pd.Series, residual_score: pd.Series,
                                residual_weight: float) -> pd.Series:
        """风格解释部分与正交残差收益预测共同构成最终股票分数。"""
        return style_score + residual_weight * residual_score

    def _validated_residual_weight(self, frame: pd.DataFrame, pca: FixedPCAStyle,
                                   features: list[str]) -> dict[str, Any]:
        """Use a purged training-period split to disable a harmful residual increment."""
        dates = pd.DatetimeIndex(frame.factor_date.drop_duplicates().sort_values())
        embargo_days = 20
        valid_position = max(embargo_days + 1, min(len(dates) - 1, int(len(dates) * .80)))
        fit_end_position = valid_position - embargo_days
        if fit_end_position < 1 or valid_position >= len(dates):
            return {"weight": 0.0, "rank_ic": np.nan, "fit_rows": 0,
                    "valid_rows": 0, "fit_end": pd.NaT, "validation_start": pd.NaT}
        fit_dates, valid_dates = dates[:fit_end_position], dates[valid_position:]
        fit = frame[frame.factor_date.isin(fit_dates)].copy()
        valid = frame[frame.factor_date.isin(valid_dates)].copy()
        selector = ResidualRidge(pca.loadings, self.config.residual_alpha).fit(
            fit, features, label="residual_label", sample_weight="sample_weight")
        evaluated = valid[["factor_date", "symbol", "residual_label"]].rename(
            columns={"residual_label": "label"})
        evaluated["prediction"] = selector.predict(valid, features)
        rank_ic = float(daily_ic(evaluated).rank_ic.mean())
        effective = self.config.residual_weight if np.isfinite(rank_ic) and rank_ic > 0 else 0.0
        return {"weight": effective, "rank_ic": rank_ic, "fit_rows": len(fit),
                "valid_rows": len(valid), "fit_end": fit_dates.max(),
                "validation_start": valid_dates.min()}

    @staticmethod
    def _component_daily_metrics(prediction: pd.DataFrame) -> pd.DataFrame:
        """在相同最终候选样本上比较风格、残差和合成分数IC。

        风格投票会把非买入候选的 ``prediction`` 设为NaN；若残差IC仍在全市场
        计算，就会与风格/最终IC使用不同股票截面，视觉上容易被误解为分数组合
        前后的损益。主组件统一使用最终候选样本，另保留全市场残差IC供审计。
        """
        frames = []
        group_columns = ["model", "pool"] if "pool" in prediction else ["model"]
        for keys, group in prediction.groupby(group_columns):
            if "pool" in prediction:
                model_name, pool = keys
            else:
                model_name = keys[0] if isinstance(keys, tuple) else keys
                pool = None
            eligible = group[group.prediction.notna()].copy()
            for column, component in (("style_timing_score", "style_timing"),
                                      ("residual_prediction_score", "orthogonal_residual"),
                                      ("prediction", "final")):
                if column in eligible:
                    metric = daily_ic(eligible, prediction=column).assign(
                        model=model_name, component=component)
                    if pool is not None: metric["pool"] = pool
                    frames.append(metric)
            if "residual_prediction_score" in group:
                metric = daily_ic(group, prediction="residual_prediction_score").assign(
                    model=model_name, component="orthogonal_residual_all_universe")
                if pool is not None: metric["pool"] = pool
                frames.append(metric)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _pca_style_rankic(contexts: dict[pd.Period, dict[str, Any]],
                          labels: pd.DataFrame,
                          styles: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Measure each frozen PCA component against next-period stock returns.

        The calculation is restricted to each prediction quarter.  It is a
        daily cross-sectional Spearman RankIC, so the report can show whether
        an individual PCA style is useful without mixing in the XGBoost timing
        forecast or its embargo-validation score.
        """
        daily_frames = []
        label_frame = labels[["factor_date", "symbol", "label"]]
        for quarter, context in contexts.items():
            exposure = context["exposure"]
            quarter_exposure = exposure[
                exposure.factor_date.dt.to_period("Q") == quarter]
            evaluated = quarter_exposure.merge(
                label_frame, on=["factor_date", "symbol"], how="left",
                validate="one_to_one")
            for style in styles:
                daily_frames.append(
                    daily_ic(evaluated, prediction=style).assign(
                        quarter=str(quarter), style=style))
        if not daily_frames:
            return pd.DataFrame(), pd.DataFrame()
        daily = pd.concat(daily_frames, ignore_index=True)
        summary = daily.groupby(["quarter", "style"], observed=True).agg(
            mean_rank_ic=("rank_ic", "mean"),
            rank_ic_std=("rank_ic", "std"),
            positive_ratio=("rank_ic", lambda values: float((values.dropna() > 0).mean())),
            observation_days=("rank_ic", "count"),
            average_stock_count=("stock_count", "mean"),
        ).reset_index()
        summary["icir"] = (
            summary.mean_rank_ic / summary.rank_ic_std.replace(0, np.nan))
        return daily, summary

    @staticmethod
    def _build_residual_target(frame: pd.DataFrame, styles: list[str]
                               ) -> tuple[pd.Series, pd.DataFrame]:
        """逐日从已实现股票收益中剔除PCA风格暴露可解释的部分。

        这里处理的是训练目标，而不是把两个预测分数直接相减。每个截面只使用
        当日暴露和最终已实现收益；后续仍由 ``label_exit_date < cutoff`` 控制可用性。
        """
        result = np.full(len(frame), np.nan, dtype=np.float64)
        audit_rows = []
        for date, indices in frame.groupby("factor_date", sort=True).indices.items():
            positions = np.asarray(indices, dtype=int)
            x = frame.iloc[positions][styles].to_numpy(dtype=np.float64, copy=False)
            y = frame.iloc[positions].label.to_numpy(dtype=np.float64, copy=False)
            valid = np.isfinite(y) & np.isfinite(x).all(axis=1)
            if valid.sum() <= len(styles) + 1:
                continue
            design = np.column_stack([np.ones(valid.sum()), x[valid]])
            coefficients = np.linalg.lstsq(design, y[valid], rcond=None)[0]
            residual = y[valid] - design @ coefficients
            result_positions = positions[valid]
            result[result_positions] = residual
            correlations = [abs(np.corrcoef(residual, x[valid, column])[0, 1])
                            for column in range(len(styles)) if x[valid, column].std() > 1e-12]
            audit_rows.append({"factor_date": date, "stock_count": int(valid.sum()),
                               "residual_std": float(residual.std(ddof=0)),
                               "max_abs_style_correlation": max(correlations, default=np.nan)})
        return pd.Series(result, index=frame.index, name="residual_label"), pd.DataFrame(audit_rows)

    def _rolling_pcas(self, processed: pd.DataFrame, features: list[str],
                      target_quarters: list[pd.Period]) -> tuple[dict[pd.Period, FixedPCAStyle],
                                                                  dict[pd.Period, dict[str, Any]]]:
        """Fit each quarter from prior complete quarters and align its components."""
        periods = processed.factor_date.dt.to_period("Q")
        pcas: dict[pd.Period, FixedPCAStyle] = {}
        metadata: dict[pd.Period, dict[str, Any]] = {}
        component_count: int | None = None
        previous: FixedPCAStyle | None = None
        if self.config.pca_fit_mode == "fixed":
            fit = processed[(processed.factor_date >= self.train_start) &
                            (processed.factor_date <= self.train_end)].copy()
            if fit.empty:
                raise ValueError("固定PCA拟合期没有有效训练数据")
            pca = FixedPCAStyle(self.config.pca_variance, self.config.pca_min_components).fit(fit, features)
            metadata = {quarter: {"prediction_quarter": str(quarter), "fit_quarters": ["fixed_training_period"],
                          "fit_start": str(fit.factor_date.min().date()), "fit_end": str(fit.factor_date.max().date()),
                          "fit_trading_days": int(fit.factor_date.nunique()), "component_count": len(pca.style_names),
                          "cumulative_explained_variance": pca.cumulative_explained_variance, "alignment": [], "mode": "fixed"}
                        for quarter in target_quarters}
            return {quarter: pca for quarter in target_quarters}, metadata
        for quarter in target_quarters:
            fit_quarters = [quarter - offset for offset in range(STYLE_PCA_WINDOW_QUARTERS, 0, -1)]
            fit = processed[periods.isin(fit_quarters)].copy()
            available_quarters = set(periods[periods.isin(fit_quarters)].unique())
            if fit.empty or not set(fit_quarters).issubset(available_quarters):
                raise ValueError(f"{quarter}前没有足够的PCA拟合数据")
            fit_end = pd.Timestamp(fit.factor_date.max())
            if fit_end >= quarter.start_time:
                raise AssertionError(f"{quarter}的PCA拟合包含目标季度数据")
            pca = FixedPCAStyle(self.config.pca_variance, self.config.pca_min_components).fit(
                fit, features, component_count=component_count)
            alignment = pd.DataFrame()
            if previous is not None:
                alignment = pca.align_to(previous)
            else:
                component_count = len(pca.style_names)
            pcas[quarter] = pca
            metadata[quarter] = {
                "prediction_quarter": str(quarter),
                "fit_quarters": [str(value) for value in fit_quarters],
                "fit_start": str(fit.factor_date.min().date()),
                "fit_end": str(fit_end.date()),
                "fit_trading_days": int(fit.factor_date.nunique()),
                "component_count": len(pca.style_names),
                "cumulative_explained_variance": pca.cumulative_explained_variance,
                "alignment": alignment.to_dict("records"),
            }
            previous = pca
        return pcas, metadata

    @staticmethod
    def _eligible_pca_quarters(processed: pd.DataFrame, start: pd.Period,
                               end: pd.Period) -> list[pd.Period]:
        """Return quarters whose complete lookback windows exist in the factor data."""
        available = set(processed.factor_date.dt.to_period("Q").unique())
        candidates = pd.period_range(start, end, freq="Q")
        return [quarter for quarter in candidates if all(
            quarter - offset in available
            for offset in range(STYLE_PCA_WINDOW_QUARTERS, 0, -1))]

    @staticmethod
    def _rolling_exposure(processed: pd.DataFrame, pcas: dict[pd.Period, FixedPCAStyle],
                          active_quarter: pd.Period, end_date: pd.Timestamp,
                          start_date: pd.Timestamp) -> pd.DataFrame:
        """Use each date's then-available PCA; future label dates use the active PCA."""
        frame = processed[(processed.factor_date >= start_date) &
                          (processed.factor_date <= end_date)]
        parts = []
        for quarter, group in frame.groupby(frame.factor_date.dt.to_period("Q"), sort=True):
            available = pcas.get(quarter) if quarter <= active_quarter else pcas[active_quarter]
            if available is None:
                continue
            parts.append(available.transform(group))
        if not parts:
            raise ValueError(f"{active_quarter}没有可用PCA暴露")
        return pd.concat(parts, ignore_index=True).sort_values(["factor_date", "symbol"])

    def _quarter_context(self, processed: pd.DataFrame, labels: pd.DataFrame,
                         features: list[str], quarter: pd.Period,
                         benchmark: pd.Series | None,
                         pca: FixedPCAStyle,
                         rolling_exposure: pd.DataFrame,
                         pool: str,
                         membership: pd.DataFrame | None) -> dict[str, Any]:
        """用当时可用的滚动PCA暴露构造季度训练和测试上下文。"""
        periods = processed.factor_date.dt.to_period("Q")

        # 训练从既定起点开始；多保留预测季度之后20个交易日用于实现样本外标签。
        quarter_dates = pd.DatetimeIndex(processed.loc[periods == quarter, "factor_date"].unique()).sort_values()
        if quarter_dates.empty:
            raise ValueError(f"{quarter}没有预测数据")
        all_dates = pd.DatetimeIndex(processed.factor_date.drop_duplicates().sort_values())
        last_position = min(len(all_dates) - 1, all_dates.get_indexer([quarter_dates.max()])[0] + 20)
        projection_end = all_dates[last_position]
        projection = processed[
            (processed.factor_date >= self.train_start) &
            (processed.factor_date <= projection_end)]
        exposure = self._inside_pool(
            rolling_exposure[rolling_exposure.factor_date <= projection_end], membership)
        if exposure.empty:
            raise ValueError(f"{pool}在{quarter}及训练期没有可用股票")
        returns = StyleReturnBuilder().build(exposure, labels, pca.style_names)
        dates = pd.DatetimeIndex(sorted(exposure.factor_date.unique()))
        builder = TimingDatasetBuilder()
        timing = builder.build_features(returns, dates, benchmark).merge(
            builder.build_labels(returns, dates), on="decision_date", how="left")
        timing = timing.merge(
            builder.build_horizon_labels(returns, dates, 5), on="decision_date", how="left")
        timing = timing.merge(
            builder.build_horizon_labels(returns, dates, 20), on="decision_date", how="left")
        timing["momentum_ic_5"] = timing.apply(
            self._cross_style_ic, axis=1, args=(pca.style_names, 5))
        timing["momentum_ic_20"] = timing.apply(
            self._cross_style_ic, axis=1, args=(pca.style_names, 20))
        timing["information_available"] = (
            timing.max_information_date.isna() |
            (timing.max_information_date <= timing.decision_date))
        if not timing.information_available.all():
            raise AssertionError("择时特征含未来信息")
        test = timing[(timing.decision_date.dt.to_period("Q") == quarter) &
                  timing.decision_date.between(self.test_start, self.test_end)].copy()
        if test.empty:
            raise ValueError(f"{quarter}没有可预测的择时样本")
        cutoff = test.decision_date.min()

        training_start = self._training_start(cutoff)
        residual_factors = projection[
            projection.factor_date.between(training_start, cutoff, inclusive="left")].copy()
        current_pca_exposure = self._inside_pool(pca.transform(residual_factors), membership)
        residual_labeled = residual_factors.merge(
            labels.loc[labels.label_exit_date < cutoff,
                       ["factor_date", "symbol", "label", "label_exit_date"]],
            on=["factor_date", "symbol"], how="inner", validate="one_to_one").merge(
            current_pca_exposure[["factor_date", "symbol", *pca.style_names]],
            on=["factor_date", "symbol"], how="inner", validate="one_to_one")
        residual_labeled["residual_label"], residual_audit = self._build_residual_target(
            residual_labeled, pca.style_names)
        metadata = {
            "prediction_quarter": str(quarter),
            "pca_mode": "rolling_previous_two_quarters",
            "pool": pool,
        }
        return {"quarter": quarter, "pca": pca,
                "pool": pool,
                "exposure": exposure, "returns": returns, "timing": timing,
                "test": test, "cutoff": cutoff, "residual_labeled": residual_labeled,
                "residual_audit": residual_audit, "metadata": metadata}

    def _apply_pca_confidence_gate(
            self, predictions: pd.DataFrame, training: pd.DataFrame,
            pca_metadata: dict[pd.Period, dict[str, Any]],
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Build point-in-time trade/retain signals without changing raw model scores."""
        output, audits = [], []
        # The setting is expressed in trading days. Predictions from a monthly
        # factor library have one observation per factor date, so using 60 raw
        # rows would mean roughly five years of history. Convert it to factor
        # periods before applying the gate.
        raw_window = self.config.confidence_rankic_window
        periods_per_year = {"daily": 252, "alternate": 126, "weekly": 52,
                            "monthly": 12, "quarterly": 4}[self.config.rebalance_frequency]
        window = max(2, int(round(raw_window * periods_per_year / 252)))
        threshold = self.config.confidence_rankic_threshold
        for pool, pool_frame in predictions.groupby("pool", sort=False):
            pool_frame = pool_frame.sort_values(["factor_date", "symbol"]).copy()
            realized = daily_ic(pool_frame.dropna(subset=["label"]))
            information = pool_frame.groupby("factor_date", as_index=False).agg(
                information_date=("label_exit_date", "max"))
            realized = realized.merge(information, on="factor_date", how="left")
            records = training[training.pool == pool].set_index("quarter")
            structural_streak = 0
            previous_quarter = None
            for date, indices in pool_frame.groupby("factor_date", sort=True).groups.items():
                date = pd.Timestamp(date)
                quarter = date.to_period("Q")
                meta = pca_metadata[quarter]
                similarities = [float(row["signed_similarity"])
                                for row in meta.get("alignment", [])]
                mean_alignment = float(np.mean(similarities)) if similarities else 1.0
                explained = float(meta["cumulative_explained_variance"])
                structural_ok = (explained >= self.config.pca_explained_threshold - 1e-12 and
                                 mean_alignment >= self.config.pca_alignment_threshold)
                if quarter != previous_quarter:
                    structural_streak = 0 if structural_ok else structural_streak + 1
                    previous_quarter = quarter
                available = realized[(realized.information_date < date) &
                                     realized.rank_ic.notna()].tail(window)
                validation_ic = float(records.loc[str(quarter), "validation_rank_ic"])
                rolling_rankic = (float(available.rank_ic.mean())
                                  if len(available) >= min(10, window) else np.nan)
                rankic_confidence = (rolling_rankic if np.isfinite(rolling_rankic)
                                     else validation_ic)
                # Warm-up quarters can precede the first leakage-free Optuna
                # split. In that case use the model score while retaining the
                # structural PCA safeguards, then switch to IC gating once a
                # validation or realized RankIC is available.
                predictive_ok = (not np.isfinite(rankic_confidence) or
                                 rankic_confidence > threshold)
                old_window = realized[(realized.information_date < date) &
                                      realized.rank_ic.notna()].tail(2 * window)
                two_bad_windows = (len(old_window) >= 2 * window and
                                   old_window.rank_ic.iloc[:window].mean() <= threshold and
                                   old_window.rank_ic.iloc[window:].mean() <= threshold)
                risk_off = structural_streak >= 2 or two_bad_windows
                trade_enabled = structural_ok and predictive_ok
                positions = np.asarray(list(indices), dtype=int)
                scores = pool_frame.loc[positions, "prediction"]
                pool_frame.loc[positions, "trade_prediction"] = (
                    scores if trade_enabled and not risk_off else np.nan)
                if risk_off:
                    pool_frame.loc[positions, "retain"] = False
                    pool_frame.loc[positions, "trade_enabled"] = True
                    mode = "risk_off_liquidate"
                elif trade_enabled:
                    pool_frame.loc[positions, "retain"] = False
                    pool_frame.loc[positions, "trade_enabled"] = True
                    mode = "normal"
                else:
                    pool_frame.loc[positions, "retain"] = True
                    pool_frame.loc[positions, "trade_enabled"] = False
                    mode = "hold"
                audits.append({"factor_date": date, "pool": pool, "quarter": str(quarter),
                               "mode": mode, "trade_enabled": bool(risk_off or trade_enabled),
                               "risk_off": risk_off, "pca_explained_variance": explained,
                               "mean_alignment": mean_alignment,
                               "structural_ok": structural_ok,
                               "structural_failure_quarters": structural_streak,
                                "available_rankic_days": len(available),
                                "confidence_window_periods": window,
                                "confidence_window_trading_days": raw_window,
                               "rolling_rankic": rolling_rankic,
                               "validation_rankic": validation_ic,
                               "rankic_confidence": rankic_confidence,
                               "predictive_ok": predictive_ok,
                               "holding_count": self.config.holding_count,
                               "rotation_quantile": self.config.rotation_quantile,
                               "rebalance_frequency": self.config.rebalance_frequency})
            output.append(pool_frame)
        result = pd.concat(output, ignore_index=True)
        result["retain"] = result.retain.astype(bool)
        result["trade_enabled"] = result.trade_enabled.astype(bool)
        return result, pd.DataFrame(audits)

    def _run_pca_stock_xgboost(self, processed: pd.DataFrame, labels: pd.DataFrame,
                               features: list[str], pcas: dict[pd.Period, FixedPCAStyle],
                               pca_metadata: dict[pd.Period, dict[str, Any]],
                               quarters: list[pd.Period], rolling_quarters: list[pd.Period],
                               effective_style_start: pd.Timestamp) -> dict[str, Any]:
        """Direct stock-level XGBoost on aligned PCA exposures, without style-return timing."""
        memberships = {pool: self._pool_membership(pool) for pool in self.pools}
        all_dates = pd.DatetimeIndex(processed.factor_date.drop_duplicates().sort_values())
        predictions, training_rows, importance_frames = [], [], []
        pca_contexts: dict[str, dict[pd.Period, dict[str, Any]]] = {
            pool: {} for pool in self.pools}
        # Tune one model on the full universe, then apply the same model to each pool.
        selections: dict[str, dict[str, Any]] = {}
        for position, (pool, quarter) in enumerate(
                [(pool, quarter) for pool in self.pools for quarter in quarters]):
            self.logger.status(
                "pca_stock_xgb", .20 + .58 * position / max(len(self.pools) * len(quarters), 1),
                f"{pool} {quarter}：PCA暴露直接训练股票XGBoost",
                model="style_rotation", pool=pool, quarter=str(quarter))
            quarter_dates = pd.DatetimeIndex(processed.loc[
                processed.factor_date.dt.to_period("Q") == quarter,
                "factor_date"].unique()).sort_values()
            if quarter_dates.empty:
                raise ValueError(f"{quarter}没有预测数据")
            rolling_exposure = self._rolling_exposure(
                processed, pcas, quarter, quarter_dates.max(), effective_style_start)
            pool_exposure = self._inside_pool(rolling_exposure, memberships[pool])
            test = pool_exposure[
                (pool_exposure.factor_date.dt.to_period("Q") == quarter) &
                pool_exposure.factor_date.between(self.test_start, self.test_end)].copy()
            if test.empty:
                raise ValueError(f"{pool} {quarter}没有PCA股票预测样本")
            cutoff = pd.Timestamp(test.factor_date.min())
            training_start = self._training_start(cutoff)
            train = rolling_exposure[
                rolling_exposure.factor_date.between(
                    training_start, cutoff, inclusive="left")].merge(
                labels.loc[labels.label_exit_date < cutoff,
                           ["factor_date", "symbol", "label", "label_exit_date"]],
                on=["factor_date", "symbol"], how="inner", validate="one_to_one")
            age = (cutoff - train.factor_date).dt.days / 365.25 * 252
            train["sample_weight"] = np.exp(
                -np.log(2) * age / STYLE_TIME_DECAY_HALFLIFE).astype(np.float32)
            selected = selections.get("global")
            model = PCAStockXGBoost(
                pcas[quarter].style_names,
                params={"max_depth": self.config.xgb_max_depth})
            if selected is None:
                # A time split needs at least one historical fit date, one
                # validation date, and one fully realized label boundary. Do
                # not launch Optuna repeatedly during the early warm-up.
                enough_history = train.factor_date.nunique() >= 4
                try:
                    if not enough_history:
                        raise ValueError("PCA股票XGB历史不足以构造无泄漏时间验证集")
                    selection = PCAStockXGBoostTuner(
                        self.logger.root / "optuna", self.optuna_trials,
                        self.logger.info, self.logger.status).tune(
                            train, pcas[quarter].style_names, self.test_year, pool)
                except ValueError as exc:
                    if "无泄漏时间验证集" not in str(exc):
                        raise
                    self.logger.info(
                        "%s %s：历史不足以进行无泄漏Optuna验证，"
                        "本期使用默认XGB参数，后续季度继续尝试调参",
                        pool, quarter)
                    model.fit(train, selected_trees=STYLE_XGB_MAX_ESTIMATORS,
                              validation_rank_ic=np.nan)
                else:
                    model = PCAStockXGBoost(pcas[quarter].style_names,
                                            params=selection["params"])
                    model.fit(train, selected_trees=selection["trees"],
                              validation_rank_ic=selection["validation_rank_ic"],
                              selection_fit_end=selection["fit_end"],
                              selection_validation_start=selection["validation_start"])
                    selections["global"] = selection
            else:
                model = PCAStockXGBoost(pcas[quarter].style_names,
                                        params=selected["params"])
                model.fit(train, selected_trees=selected["trees"],
                          validation_rank_ic=selected["validation_rank_ic"],
                          selection_fit_end=selected["fit_end"],
                          selection_validation_start=selected["validation_start"])
            prediction = test.merge(
                labels[["factor_date", "symbol", "label", "label_exit_date"]],
                on=["factor_date", "symbol"], how="left", validate="one_to_one")
            prediction["prediction"] = model.predict(prediction)
            prediction["model"], prediction["pool"] = "style_rotation", pool
            predictions.append(prediction)
            train_evaluated = train[["factor_date", "symbol", "label"]].copy()
            train_evaluated["prediction"] = model.predict(train)
            train_rank_ic = float(daily_ic(train_evaluated).rank_ic.mean())
            quarter_rank_ic = float(daily_ic(prediction.dropna(subset=["label"])).rank_ic.mean())
            training_rows.append({
                "model": "style_rotation", "pool": pool, "quarter": str(quarter),
                "train_start": str(train.factor_date.min()),
                "train_end": str(train.label_exit_date.max()),
                "train_samples": len(train), "test_samples": len(test),
                "feature_count": len(pcas[quarter].style_names),
                "xgb_max_depth": model.params.get("max_depth"),
                "xgb_params": json.dumps(model.params, ensure_ascii=False),
                "best_trees": model.best_trees,
                "validation_rank_ic": model.validation_rank_ic,
                "train_rank_ic": train_rank_ic, "quarter_rank_ic": quarter_rank_ic})
            importance_frames.append(model.feature_importance_frame().assign(
                pool=pool, quarter=str(quarter)))
            pca_contexts[pool][quarter] = {"exposure": test}

        prediction = pd.concat(predictions, ignore_index=True)
        training_frame = pd.DataFrame(training_rows)
        prediction, confidence_audit = self._apply_pca_confidence_gate(
            prediction, training_frame, pca_metadata)
        daily = pd.concat([
            daily_ic(group).assign(model="style_rotation", pool=pool)
            for pool, group in prediction.groupby("pool")], ignore_index=True)
        summary = pd.concat([
            layered_metrics(group).assign(pool=pool)
            for pool, group in prediction.groupby("pool")], ignore_index=True)
        pca_daily_parts, pca_summary_parts = [], []
        styles = pcas[quarters[0]].style_names
        for pool in self.pools:
            component_daily, component_summary = self._pca_style_rankic(
                pca_contexts[pool], labels, styles)
            pca_daily_parts.append(component_daily.assign(pool=pool))
            pca_summary_parts.append(component_summary.assign(pool=pool))
        pca_style_daily = pd.concat(pca_daily_parts, ignore_index=True)
        pca_style_summary = pd.concat(pca_summary_parts, ignore_index=True)
        loading_frame = pd.concat([
            pcas[quarter].loading_frame().assign(quarter=str(quarter))
            for quarter in rolling_quarters], ignore_index=True)
        self.logger.csv("style/pca_loadings.csv", loading_frame)
        self.logger.json("style/pca_metadata.json", {
            "mode": "rolling_previous_two_quarters_direct_stock_xgboost",
            "window_quarters": STYLE_PCA_WINDOW_QUARTERS,
            "requested_training_start": str(self.train_start.date()),
            "effective_style_training_start": str(effective_style_start.date()),
            "target_cumulative_explained_variance": self.config.pca_variance,
            "quarters": [pca_metadata[quarter] for quarter in rolling_quarters]})
        prediction.to_parquet(
            self.logger.root / "predictions/all_predictions.parquet", index=False)
        for pool, group in prediction.groupby("pool"):
            target = (self.logger.root / "predictions/model_prediction_files" /
                      f"style_rotation_{pool}_predictions.parquet")
            group[["factor_date", "symbol", "prediction"]].rename(
                columns={"factor_date": "date", "symbol": "ticker",
                         "prediction": "style_rotation"}).set_index(
                ["ticker", "date"]).sort_index().to_parquet(target)
        self.logger.csv("timing/xgb_feature_importance.csv",
                        pd.concat(importance_frames, ignore_index=True))
        self.logger.csv("audit/pca_confidence_gate.csv", confidence_audit)
        self.logger.csv("timing/style_weights.csv", pd.DataFrame())
        self.logger.csv("timing/consensus_votes.csv", pd.DataFrame())
        self.logger.csv("metrics/daily_rankic.csv", daily)
        self.logger.csv("metrics/score_component_rankic.csv",
                        daily.assign(component="pca_stock_xgboost"))
        self.logger.csv("metrics/pca_style_daily_rankic.csv", pca_style_daily)
        self.logger.csv("metrics/pca_style_rankic.csv", pca_style_summary)
        self.logger.csv("metrics/summary.csv", summary)
        self.logger.csv("metrics/training_records.csv", training_frame)
        return {"predictions": prediction, "daily_metrics": daily, "summary": summary,
                "training_records": training_frame, "weights": pd.DataFrame(),
                "vote_audit": pd.DataFrame(), "pca_style_metrics": pca_style_summary}

    def run(self) -> dict[str, Any]:
        self.logger.status("style_load", .03, "读取因子与价格数据")
        first_pca_fit_start = self.train_start.to_period("Q") - STYLE_PCA_WINDOW_QUARTERS
        load_start = min(self.train_start, first_pca_fit_start.start_time)
        bundle = ExistingDataAdapter(
            self.factor_path, selected_features=self.selected_features).load(
            str(load_start.date()), str(self.test_end.date()), "ALL_MARKET" in self.pools)
        features = ExistingDataAdapter.feature_columns(bundle.factors)
        labels = LabelBuilder().build(bundle.real_twap, bundle.factors.factor_date.unique())
        self.logger.csv("audit/data_quality.csv", bundle.quality)

        self.logger.status("style_preprocess", .12, "用前两个完整季度滚动拟合PCA并对齐载荷")
        processed = CrossSectionalPreprocessor().transform(bundle.factors, features)
        quarters = list(pd.period_range(self.test_start, self.test_end, freq="Q"))
        first_quarter = self.train_start.to_period("Q")
        rolling_quarters = (list(pd.period_range(first_quarter, quarters[-1], freq="Q"))
                            if self.config.pca_fit_mode == "fixed" else
                            self._eligible_pca_quarters(processed, first_quarter, quarters[-1]))
        missing_prediction_quarters = sorted(set(quarters) - set(rolling_quarters))
        if missing_prediction_quarters:
            raise ValueError(
                "测试季度缺少PCA历史窗口：" + ", ".join(map(str, missing_prediction_quarters)))
        if not rolling_quarters:
            raise ValueError("因子数据没有任何具备前两个完整季度历史的PCA拟合季度")
        effective_style_start = max(self.train_start, rolling_quarters[0].start_time)
        pcas, pca_metadata = self._rolling_pcas(processed, features, rolling_quarters)
        if self.models == ["pca_stock_xgboost"]:
            return self._run_pca_stock_xgboost(
                processed, labels, features, pcas, pca_metadata, quarters,
                rolling_quarters, effective_style_start)
        contexts: dict[tuple[str, pd.Period], dict[str, Any]] = {}
        memberships = {pool: self._pool_membership(pool) for pool in self.pools}
        all_dates = pd.DatetimeIndex(processed.factor_date.drop_duplicates().sort_values())
        total_contexts = max(1, len(self.pools) * len(quarters))
        context_number = 0
        for pool in self.pools:
            benchmark = self._benchmark(self._pool_benchmark_path(pool))
            for quarter in quarters:
                self.logger.status(
                    "style_returns", .18 + .12 * context_number / total_contexts,
                    f"{pool} {quarter}：使用共享PCA坐标构造池内训练样本")
                context_number += 1
                quarter_dates = pd.DatetimeIndex(
                    processed.loc[processed.factor_date.dt.to_period("Q") == quarter,
                                  "factor_date"].unique()).sort_values()
                if quarter_dates.empty:
                    raise ValueError(f"{quarter}没有预测数据")
                last_position = min(len(all_dates) - 1,
                                    all_dates.get_indexer([quarter_dates.max()])[0] + 20)
                rolling_exposure = self._rolling_exposure(
                    processed, pcas, quarter, all_dates[last_position], effective_style_start)
                contexts[(pool, quarter)] = self._quarter_context(
                    processed, labels, features, quarter, benchmark, pcas[quarter],
                    rolling_exposure, pool, memberships[pool])
        pca = pcas[quarters[0]]
        pca_daily_parts, pca_summary_parts = [], []
        for pool in self.pools:
            pool_contexts = {quarter: contexts[(pool, quarter)] for quarter in quarters}
            pool_daily, pool_summary = self._pca_style_rankic(
                pool_contexts, labels, pca.style_names)
            pca_daily_parts.append(pool_daily.assign(pool=pool))
            pca_summary_parts.append(pool_summary.assign(pool=pool))
        pca_style_daily = pd.concat(pca_daily_parts, ignore_index=True)
        pca_style_summary = pd.concat(pca_summary_parts, ignore_index=True)
        loading_frames = []
        for quarter in rolling_quarters:
            loading_frames.append(pcas[quarter].loading_frame().assign(quarter=str(quarter)))
        self.logger.csv("style/pca_loadings.csv", pd.concat(loading_frames, ignore_index=True))
        self.logger.json("style/pca_metadata.json", {
            "mode": "rolling_previous_two_quarters",
            "window_quarters": STYLE_PCA_WINDOW_QUARTERS,
            "requested_training_start": str(self.train_start.date()),
            "effective_style_training_start": str(effective_style_start.date()),
            "target_cumulative_explained_variance": self.config.pca_variance,
            "quarters": [pca_metadata[quarter] for quarter in rolling_quarters]})
        self.logger.csv("style/style_returns.csv", pd.concat(
            [context["returns"].assign(pool=context["pool"])
             for context in contexts.values()], ignore_index=True
        ).drop_duplicates(["pool", "factor_date", "style"]).sort_values(["pool", "factor_date"]))
        self.logger.csv("audit/timing_alignment.csv", pd.concat([
            context["timing"][["decision_date", "max_information_date", "label_exit_date",
                               "information_available"]].assign(pool=context["pool"])
            for context in contexts.values()], ignore_index=True
        ).drop_duplicates(["pool", "decision_date"]).sort_values(["pool", "decision_date"]))
        self.logger.csv("audit/momentum_baseline_ic.csv", pd.concat([
            context["timing"][["decision_date", "label_exit_date_5", "label_exit_date_20",
                               "momentum_ic_5", "momentum_ic_20"]].assign(
                                   prediction_quarter=str(quarter), pool=pool)
            for (pool, quarter), context in contexts.items()], ignore_index=True))
        self.logger.csv("audit/residual_target_orthogonality.csv", pd.concat([
            context["residual_audit"].assign(prediction_quarter=str(quarter), pool=pool)
            for (pool, quarter), context in contexts.items()], ignore_index=True))

        predictions, weight_frames, training_rows, vote_frames = [], [], [], []
        residual_coefficients = []
        xgb_feature_importance = []
        style_forecast_audit = []
        for model_name in self.models:
            output_name = "style_rotation" if model_name == "xgboost_dual_horizon" else f"style_{model_name}"
            model_weights = []
            if model_name == "xgboost_dual_horizon":
                model_scores, model_votes = [], []
                # 20日隔离只用于首次（2022—2023）参数选择；2024各季度固定树数，
                # 用截止当季前全部已实现标签重训，不再重复损失20个交易日样本。
                xgb_selection: dict[str, dict[int, dict[str, Any]]] = {}
                forecast_ic_history: dict[str, list[pd.DataFrame]] = {}
                for training_index, ((pool, quarter), context) in enumerate(contexts.items()):
                    self.logger.status(
                        "style_training", .35 + .40 * training_index / max(len(contexts), 1),
                        f"{pool} {quarter}：滚动训练双期限XGBoost与残差模型",
                        model=output_name, pool=pool, quarter=str(quarter))
                    pca, timing, test, cutoff = (context["pca"], context["timing"],
                                                  context["test"], context["cutoff"])
                    training_start = self._training_start(cutoff)
                    train5 = timing[(timing.decision_date >= training_start) &
                                    (timing.label_exit_date_5 < cutoff) &
                                    timing[f"target_{pca.style_names[0]}_5"].notna()].copy()
                    train20 = timing[(timing.decision_date >= training_start) &
                                     (timing.label_exit_date_20 < cutoff) &
                                     timing[f"target_{pca.style_names[0]}_20"].notna()].copy()
                    for train in (train5, train20):
                        age = (cutoff - train.decision_date).dt.days / 365.25 * 252
                        train["sample_weight"] = np.exp(-np.log(2) * age / STYLE_TIME_DECAY_HALFLIFE)
                    pool_selection = xgb_selection.setdefault(pool, {})
                    pool_forecast_history = forecast_ic_history.setdefault(pool, [])
                    selection_was_reused = bool(pool_selection)
                    models_by_horizon = {}
                    for horizon, train in ((5, train5), (20, train20)):
                        model = DualHorizonXGBoost(
                            pca.style_names, horizon,
                            params={"max_depth": self.config.xgb_max_depth})
                        if horizon not in pool_selection:
                            model.fit(train)
                            pool_selection[horizon] = {
                                "trees": model.best_iterations[0],
                                "validation_rank_ic": model.validation_rank_ic,
                                "fit_end": model.fit_end_,
                                "validation_start": model.validation_start_}
                        else:
                            selected = pool_selection[horizon]
                            model.fit(train, selected_trees=selected["trees"],
                                      validation_rank_ic=selected["validation_rank_ic"],
                                      selection_fit_end=selected["fit_end"],
                                      selection_validation_start=selected["validation_start"])
                        models_by_horizon[horizon] = model
                    model5, model20 = models_by_horizon[5], models_by_horizon[20]
                    forecast5, forecast20 = model5.predict(test), model20.predict(test)
                    quarter_forecast_ic = pd.concat([
                        self._daily_forecast_ic(forecast5, test, pca.style_names, 5),
                        self._daily_forecast_ic(forecast20, test, pca.style_names, 20)],
                        ignore_index=True)
                    available_forecast_ic = pd.concat(
                        [*pool_forecast_history, quarter_forecast_ic], ignore_index=True)
                    train_ic5 = self._mean_forecast_ic(model5.predict(train5), train5, pca.style_names, 5)
                    train_ic20 = self._mean_forecast_ic(model20.predict(train20), train20, pca.style_names, 20)
                    oos_ic5 = self._mean_forecast_ic(forecast5, test, pca.style_names, 5)
                    oos_ic20 = self._mean_forecast_ic(forecast20, test, pca.style_names, 20)
                    for horizon, fitted in ((5, model5), (20, model20)):
                        importance = fitted.feature_importance_frame()
                        importance["quarter"], importance["horizon"], importance["pool"] = (
                            str(quarter), horizon, pool)
                        xgb_feature_importance.append(importance)
                    for horizon, forecast_by_horizon in ((5, forecast5), (20, forecast20)):
                        predicted_long = forecast_by_horizon.melt(
                            id_vars="decision_date", var_name="style", value_name="predicted_return")
                        target_names = [f"target_{style}_{horizon}" for style in pca.style_names]
                        realized_long = test[["decision_date", f"label_exit_date_{horizon}",
                                              *target_names]].melt(
                            id_vars=["decision_date", f"label_exit_date_{horizon}"],
                            var_name="target_name", value_name="realized_return")
                        realized_long["style"] = realized_long.target_name.str.replace(
                            rf"^target_|_{horizon}$", "", regex=True)
                        aligned = predicted_long.merge(
                            realized_long.drop(columns="target_name"),
                            on=["decision_date", "style"], how="left", validate="one_to_one")
                        aligned["quarter"], aligned["horizon"], aligned["pool"] = (
                            str(quarter), horizon, pool)
                        style_forecast_audit.append(aligned)
                    dynamic = self._dynamic_horizon_weights(timing, test, pca.style_names,
                        model5.validation_rank_ic, model20.validation_rank_ic,
                        available_forecast_ic)
                    pool_forecast_history.append(quarter_forecast_ic)
                    combined = dynamic.merge(forecast5, on="decision_date").merge(
                        forecast20, on="decision_date", suffixes=("_5", "_20"))
                    forecast = combined[["decision_date"]].copy()
                    for style in pca.style_names:
                        forecast[style] = combined.weight_5 * combined[f"{style}_5"] + combined.weight_20 * combined[f"{style}_20"]
                    quarter_exposure = context["exposure"][
                        context["exposure"].factor_date.isin(test.decision_date)]
                    score, votes = self._consensus_stock_scores(
                        quarter_exposure, forecast, pca.style_names,
                        self.config.buy_confirmations, self.config.vote_quantile)
                    residual_train = context["residual_labeled"][
                        context["residual_labeled"].residual_label.notna()].copy()
                    age = (cutoff - residual_train.factor_date).dt.days / 365.25 * 252
                    residual_train["sample_weight"] = np.exp(
                        -np.log(2) * age / STYLE_TIME_DECAY_HALFLIFE).astype(np.float32)
                    residual_validation = self._validated_residual_weight(
                        residual_train, pca, features)
                    residual_model = ResidualRidge(pca.loadings, self.config.residual_alpha).fit(
                        residual_train, features, label="residual_label", sample_weight="sample_weight")
                    if residual_model.max_label_exit_date_ is not None and residual_model.max_label_exit_date_ >= cutoff:
                        raise AssertionError("残差模型训练包含尚未实现的未来标签")
                    quarter_keys = quarter_exposure[["factor_date", "symbol"]]
                    quarter_factors = processed.merge(
                        quarter_keys, on=["factor_date", "symbol"], how="inner", validate="one_to_one")
                    residual_score = self._residual_score(residual_model, quarter_factors, features)
                    score = score.rename(columns={"prediction": "style_timing_score"}).merge(
                        residual_score, on=["factor_date", "symbol"], how="left", validate="one_to_one")
                    # 残差模型预测的是风格无法解释的收益增量，因此与风格分数相加。
                    # NaN风格分数仍为非买入候选，不由残差模型强行补仓。
                    score["prediction"] = self._combine_style_residual(
                        score.style_timing_score, score.residual_prediction_score,
                        residual_validation["weight"])
                    score = score.merge(
                        votes[["factor_date", "symbol", "buy_votes", "sell_votes", "buy_eligible"]],
                        on=["factor_date", "symbol"], how="left", validate="one_to_one")
                    risk = dynamic[["decision_date", "risk_off"]].rename(
                        columns={"decision_date": "factor_date"})
                    score = score.merge(risk, on="factor_date", how="left", validate="many_to_one")
                    coefficient = residual_model.coefficient_frame(features)
                    coefficient["quarter"] = str(quarter)
                    coefficient["pool"] = pool
                    coefficient["train_end"] = residual_model.max_label_exit_date_
                    residual_coefficients.append(coefficient)
                    score["pool"], votes["pool"] = pool, pool
                    model_scores.append(score); model_votes.append(votes)
                    dynamic["model"], dynamic["pool"] = output_name, pool
                    model_weights.append(dynamic)
                    training_rows.append({"model": output_name, "pool": pool, "quarter": str(quarter),
                        "train_end": str(max(train5.label_exit_date_5.max(), train20.label_exit_date_20.max())),
                        "train_start_5": str(train5.decision_date.min()), "train_end_5": str(train5.label_exit_date_5.max()),
                        "train_start_20": str(train20.decision_date.min()), "train_end_20": str(train20.label_exit_date_20.max()),
                        "train_samples_5": len(train5), "train_samples_20": len(train20),
                        "train_samples": len(train5) + len(train20), "test_days": len(test),
                        "residual_train_samples": residual_model.train_rows_,
                        "residual_train_end": str(residual_model.max_label_exit_date_),
                        "residual_alpha": self.config.residual_alpha,
                        "residual_weight": self.config.residual_weight,
                        "effective_residual_weight": residual_validation["weight"],
                        "residual_validation_rank_ic": residual_validation["rank_ic"],
                        "residual_validation_fit_rows": residual_validation["fit_rows"],
                        "residual_validation_rows": residual_validation["valid_rows"],
                        "residual_validation_fit_end": str(residual_validation["fit_end"]),
                        "residual_validation_start": str(residual_validation["validation_start"]),
                        "train_rank_ic_5": train_ic5, "train_rank_ic_20": train_ic20,
                        "validation_rank_ic_5": model5.validation_rank_ic,
                        "validation_rank_ic_20": model20.validation_rank_ic,
                        "oos_style_rank_ic_5": oos_ic5,
                        "oos_style_rank_ic_20": oos_ic20,
                        "selection_fit_end_5": str(model5.fit_end_),
                        "selection_validation_start_5": str(model5.validation_start_),
                        "selection_fit_end_20": str(model20.fit_end_),
                        "selection_validation_start_20": str(model20.validation_start_),
                        "selection_embargo_days": model5.embargo_days_,
                        "xgb_max_depth": self.config.xgb_max_depth,
                        "selection_reused": selection_was_reused,
                        "validation_rank_ic": float(dynamic.weight_5.mean() * model5.validation_rank_ic +
                                                    dynamic.weight_20.mean() * model20.validation_rank_ic),
                        "best_trees_5": int(np.median(model5.best_iterations)),
                        "best_trees_20": int(np.median(model20.best_iterations)),
                        "train_rank_ic": float(dynamic.weight_5.mean() * train_ic5 +
                                               dynamic.weight_20.mean() * train_ic20)})
                score = pd.concat(model_scores, ignore_index=True)
                score = score.merge(labels[["factor_date", "symbol", "label"]], on=["factor_date", "symbol"], how="left")
                score["model"] = output_name; predictions.append(score)
                votes = pd.concat(model_votes, ignore_index=True); votes["model"] = output_name; vote_frames.append(votes)
                weight_frames.append(pd.concat(model_weights, ignore_index=True))
                continue
            model_scores = []
            for (pool, quarter), context in contexts.items():
                pca, timing, test, cutoff = (context["pca"], context["timing"],
                                              context["test"], context["cutoff"])
                training_start = self._training_start(cutoff)
                train = timing[(timing.decision_date >= training_start) &
                               (timing.label_exit_date < cutoff) & timing.best_style.notna()].copy()
                age = (cutoff - train.decision_date).dt.days / 365.25 * 252
                train["sample_weight"] = np.exp(-np.log(2) * age / STYLE_TIME_DECAY_HALFLIFE) * (1 + train.winner_margin.clip(lower=0))
                timer = StyleTimer(model_name, pca.style_names).fit(train)
                weight = timer.predict_weights(test)
                weight["pool"] = pool
                model_weights.append(weight)
                training_rows.append({"model": output_name, "pool": pool, "quarter": str(quarter), "train_end": str(train.label_exit_date.max()),
                                      "train_samples": len(train), "test_days": len(test)})
                quarter_exposure = context["exposure"][
                    context["exposure"].factor_date.isin(test.decision_date)]
                model_scores.append(self._stock_scores(
                    quarter_exposure, weight, pca.style_names).assign(pool=pool))
            weights = pd.concat(model_weights, ignore_index=True); weights["model"] = output_name
            weight_frames.append(weights)
            score = pd.concat(model_scores, ignore_index=True)
            score = score.merge(labels[["factor_date", "symbol", "label"]], on=["factor_date", "symbol"], how="left")
            score["model"] = output_name; predictions.append(score)
        prediction = pd.concat(predictions, ignore_index=True)
        weights = pd.concat(weight_frames, ignore_index=True)
        vote_audit = pd.concat(vote_frames, ignore_index=True) if vote_frames else pd.DataFrame()
        daily = pd.concat([
            daily_ic(group).assign(model=model_name, pool=pool)
            for (model_name, pool), group in prediction.groupby(["model", "pool"])
        ], ignore_index=True)
        component_daily_frame = self._component_daily_metrics(prediction)
        summary = pd.concat([
            layered_metrics(group).assign(pool=pool)
            for pool, group in prediction.groupby("pool")
        ], ignore_index=True)
        training_frame = pd.DataFrame(training_rows)
        if not training_frame.empty:
            quarterly = daily.assign(quarter=daily.factor_date.dt.to_period("Q").astype(str)).groupby(
                ["model", "pool", "quarter"], observed=True).rank_ic.agg(["mean", "std"]).reset_index()
            quarterly["quarter_icir"] = quarterly["mean"] / quarterly["std"].replace(0, np.nan)
            training_frame = training_frame.merge(
                quarterly.rename(columns={"mean": "quarter_rank_ic"})[
                    ["model", "pool", "quarter", "quarter_rank_ic", "quarter_icir"]],
                on=["model", "pool", "quarter"], how="left")
            if "train_rank_ic" not in training_frame:
                training_frame["train_rank_ic"] = np.nan
        # 完整逐股票预测与其他模型一致使用Parquet；避免重复写百万行CSV。
        prediction.to_parquet(self.logger.root / "predictions/all_predictions.parquet", index=False)
        for (model_name, pool), group in prediction.groupby(["model", "pool"]):
            target = (self.logger.root / "predictions/model_prediction_files" /
                      f"{model_name}_{pool}_predictions.parquet")
            factor_file = group[["factor_date", "symbol", "prediction"]].rename(
                columns={"factor_date": "date", "symbol": "ticker", "prediction": model_name}
            ).set_index(["ticker", "date"]).sort_index()
            factor_file.to_parquet(target)
        self.logger.csv("timing/style_weights.csv", weights)
        self.logger.csv("timing/consensus_votes.csv", vote_audit)
        self.logger.csv("style/residual_ridge_coefficients.csv",
                        pd.concat(residual_coefficients, ignore_index=True) if residual_coefficients else pd.DataFrame())
        self.logger.csv("timing/xgb_feature_importance.csv",
                        pd.concat(xgb_feature_importance, ignore_index=True)
                        if xgb_feature_importance else pd.DataFrame())
        self.logger.csv("audit/style_forecast_alignment.csv",
                        pd.concat(style_forecast_audit, ignore_index=True)
                        if style_forecast_audit else pd.DataFrame())
        self.logger.csv("metrics/daily_rankic.csv", daily)
        self.logger.csv("metrics/score_component_rankic.csv", component_daily_frame)
        self.logger.csv("metrics/pca_style_daily_rankic.csv", pca_style_daily)
        self.logger.csv("metrics/pca_style_rankic.csv", pca_style_summary)
        self.logger.csv("metrics/summary.csv", summary)
        self.logger.csv("metrics/training_records.csv", training_frame)
        return {"predictions": prediction, "daily_metrics": daily, "summary": summary,
                "training_records": training_frame, "weights": weights,
                "vote_audit": vote_audit,
                "pca_style_metrics": pca_style_summary}
