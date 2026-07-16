from imports import *
from env import (STYLE_MODELS, STYLE_PCA_END, STYLE_PCA_START,
                 STYLE_TIME_DECAY_HALFLIFE, STYLE_TIMING_TRAIN_START)
from rolling_ml.data_loader import ExistingDataAdapter
from rolling_ml.label_builder import LabelBuilder
from rolling_ml.metrics import daily_ic, layered_metrics
from style_rotation.models import DualHorizonXGBoost, StyleTimer
from style_rotation.config import StyleRuntimeConfig
from style_rotation.pca_style import FixedPCAStyle
from style_rotation.preprocessing import CrossSectionalPreprocessor
from style_rotation.residual_model import ResidualRidge
from style_rotation.style_returns import StyleReturnBuilder
from style_rotation.timing import TimingDatasetBuilder


class StyleRotationPipeline:
    """端到端风格工程。输出仍是BackTestAdapter要求的t0股票连续分数。"""

    def __init__(self, logger, test_year: int = 2024, models: list[str] | None = None,
                 config: StyleRuntimeConfig | None = None):
        self.logger, self.test_year = logger, test_year
        self.models = models or list(STYLE_MODELS)
        self.config = (config or StyleRuntimeConfig()).validate()

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

    @staticmethod
    def _component_daily_metrics(prediction: pd.DataFrame) -> pd.DataFrame:
        """在相同最终候选样本上比较风格、残差和合成分数IC。

        风格投票会把非买入候选的 ``prediction`` 设为NaN；若残差IC仍在全市场
        计算，就会与风格/最终IC使用不同股票截面，视觉上容易被误解为分数组合
        前后的损益。主组件统一使用最终候选样本，另保留全市场残差IC供审计。
        """
        frames = []
        for model_name, group in prediction.groupby("model"):
            eligible = group[group.prediction.notna()].copy()
            for column, component in (("style_timing_score", "style_timing"),
                                      ("residual_prediction_score", "orthogonal_residual"),
                                      ("prediction", "final")):
                if column in eligible:
                    frames.append(daily_ic(eligible, prediction=column).assign(
                        model=model_name, component=component))
            if "residual_prediction_score" in group:
                frames.append(daily_ic(group, prediction="residual_prediction_score").assign(
                    model=model_name, component="orthogonal_residual_all_universe"))
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

    @staticmethod
    def _fixed_pca_sample(processed: pd.DataFrame) -> pd.DataFrame:
        """Return the one historical PCA fit sample; no prediction-period row is allowed."""
        fit = processed[processed.factor_date.between(
            pd.Timestamp(STYLE_PCA_START), pd.Timestamp(STYLE_PCA_END))].copy()
        if fit.empty:
            raise ValueError(f"固定PCA拟合区间没有数据：{STYLE_PCA_START} 至 {STYLE_PCA_END}")
        if fit.factor_date.max() > pd.Timestamp(STYLE_PCA_END):
            raise AssertionError("固定PCA拟合包含拟合结束日之后的数据")
        return fit

    def _quarter_context(self, processed: pd.DataFrame, labels: pd.DataFrame,
                         features: list[str], quarter: pd.Period,
                         benchmark: pd.Series | None,
                         pca: FixedPCAStyle) -> dict[str, Any]:
        """使用同一套历史期固定PCA载荷构造一个预测季度的训练和测试上下文。"""
        periods = processed.factor_date.dt.to_period("Q")

        # 训练从既定起点开始；多保留预测季度之后20个交易日用于实现样本外标签。
        quarter_dates = pd.DatetimeIndex(processed.loc[periods == quarter, "factor_date"].unique()).sort_values()
        if quarter_dates.empty:
            raise ValueError(f"{quarter}没有预测数据")
        all_dates = pd.DatetimeIndex(processed.factor_date.drop_duplicates().sort_values())
        last_position = min(len(all_dates) - 1, all_dates.get_indexer([quarter_dates.max()])[0] + 20)
        projection_end = all_dates[last_position]
        projection = processed[
            (processed.factor_date >= pd.Timestamp(STYLE_TIMING_TRAIN_START)) &
            (processed.factor_date <= projection_end)]
        exposure = pca.transform(projection)
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
        test = timing[timing.decision_date.dt.to_period("Q") == quarter].copy()
        if test.empty:
            raise ValueError(f"{quarter}没有可预测的择时样本")
        cutoff = test.decision_date.min()

        residual_labeled = projection[projection.factor_date < cutoff].merge(
            labels.loc[labels.label_exit_date < cutoff,
                       ["factor_date", "symbol", "label", "label_exit_date"]],
            on=["factor_date", "symbol"], how="inner", validate="one_to_one").merge(
            exposure[["factor_date", "symbol", *pca.style_names]],
            on=["factor_date", "symbol"], how="inner", validate="one_to_one")
        residual_labeled["residual_label"], residual_audit = self._build_residual_target(
            residual_labeled, pca.style_names)
        metadata = {
            "prediction_quarter": str(quarter),
            "pca_mode": "fixed_historical_once",
            "pca_fit_start": STYLE_PCA_START,
            "pca_fit_end": STYLE_PCA_END,
        }
        return {"quarter": quarter, "pca": pca,
                "exposure": exposure, "returns": returns, "timing": timing,
                "test": test, "cutoff": cutoff, "residual_labeled": residual_labeled,
                "residual_audit": residual_audit, "metadata": metadata}

    def run(self) -> dict[str, Any]:
        self.logger.status("style_load", .03, "读取因子与价格数据")
        bundle = ExistingDataAdapter().load(STYLE_PCA_START, f"{self.test_year}-12-31")
        features = ExistingDataAdapter.feature_columns(bundle.factors)
        labels = LabelBuilder().build(bundle.real_twap)
        self.logger.csv("audit/data_quality.csv", bundle.quality)

        self.logger.status("style_preprocess", .12, "用历史区间拟合一次PCA并冻结载荷")
        processed = CrossSectionalPreprocessor().transform(bundle.factors, features)
        fit = self._fixed_pca_sample(processed)
        test_start = pd.Timestamp(f"{self.test_year}-01-01")
        if fit.factor_date.max() >= test_start:
            raise AssertionError("固定PCA拟合区间包含测试期数据")
        pca = FixedPCAStyle(self.config.pca_variance, self.config.pca_min_components).fit(
            fit, features)
        from env import DATA_DIR
        benchmark = self._benchmark(DATA_DIR / "Benchmark_zz1000.parquet")
        quarters = list(pd.period_range(f"{self.test_year}Q1", f"{self.test_year}Q4", freq="Q"))
        contexts = {}
        for position, quarter in enumerate(quarters):
            self.logger.status("style_returns", .18 + .12 * position,
                               f"{quarter}：沿用固定PCA载荷构造滚动训练样本")
            contexts[quarter] = self._quarter_context(
                processed, labels, features, quarter, benchmark, pca)
        pca_style_daily, pca_style_summary = self._pca_style_rankic(
            contexts, labels, pca.style_names)
        self.logger.csv("style/pca_loadings.csv", pca.loading_frame())
        self.logger.json("style/pca_metadata.json", {
            "mode": "fixed_historical_once",
            "fit_start": str(fit.factor_date.min().date()),
            "fit_end": str(fit.factor_date.max().date()),
            "fit_trading_days": int(fit.factor_date.nunique()),
            "aggregation": "mean_daily_cross_sectional_correlation",
            "component_count": len(pca.style_names),
            "target_cumulative_explained_variance": self.config.pca_variance,
            "achieved_cumulative_explained_variance": pca.cumulative_explained_variance,
            "explained_variance": pca.explained_variance.tolist(),
            "quarters": [context["metadata"] for context in contexts.values()]})
        self.logger.csv("style/style_returns.csv", pd.concat(
            [context["returns"] for context in contexts.values()], ignore_index=True
        ).drop_duplicates("factor_date").sort_values("factor_date"))
        self.logger.csv("audit/timing_alignment.csv", pd.concat([
            context["timing"][["decision_date", "max_information_date", "label_exit_date",
                               "information_available"]]
            for context in contexts.values()], ignore_index=True
        ).drop_duplicates("decision_date").sort_values("decision_date"))
        self.logger.csv("audit/momentum_baseline_ic.csv", pd.concat([
            context["timing"][["decision_date", "label_exit_date_5", "label_exit_date_20",
                               "momentum_ic_5", "momentum_ic_20"]].assign(
                                   prediction_quarter=str(quarter))
            for quarter, context in contexts.items()], ignore_index=True))
        self.logger.csv("audit/residual_target_orthogonality.csv", pd.concat([
            context["residual_audit"].assign(prediction_quarter=str(quarter))
            for quarter, context in contexts.items()], ignore_index=True))

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
                xgb_selection: dict[int, dict[str, Any]] = {}
                forecast_ic_history: list[pd.DataFrame] = []
                for quarter, context in contexts.items():
                    pca, timing, test, cutoff = (context["pca"], context["timing"],
                                                  context["test"], context["cutoff"])
                    train5 = timing[(timing.decision_date >= STYLE_TIMING_TRAIN_START) &
                                    (timing.label_exit_date_5 < cutoff) &
                                    timing[f"target_{pca.style_names[0]}_5"].notna()].copy()
                    train20 = timing[(timing.decision_date >= STYLE_TIMING_TRAIN_START) &
                                     (timing.label_exit_date_20 < cutoff) &
                                     timing[f"target_{pca.style_names[0]}_20"].notna()].copy()
                    for train in (train5, train20):
                        age = (cutoff - train.decision_date).dt.days / 365.25 * 252
                        train["sample_weight"] = np.exp(-np.log(2) * age / STYLE_TIME_DECAY_HALFLIFE)
                    selection_was_reused = bool(xgb_selection)
                    models_by_horizon = {}
                    for horizon, train in ((5, train5), (20, train20)):
                        model = DualHorizonXGBoost(pca.style_names, horizon)
                        if horizon not in xgb_selection:
                            model.fit(train)
                            xgb_selection[horizon] = {
                                "trees": model.best_iterations[0],
                                "validation_rank_ic": model.validation_rank_ic,
                                "fit_end": model.fit_end_,
                                "validation_start": model.validation_start_}
                        else:
                            selected = xgb_selection[horizon]
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
                        [*forecast_ic_history, quarter_forecast_ic], ignore_index=True)
                    train_ic5 = self._mean_forecast_ic(model5.predict(train5), train5, pca.style_names, 5)
                    train_ic20 = self._mean_forecast_ic(model20.predict(train20), train20, pca.style_names, 20)
                    oos_ic5 = self._mean_forecast_ic(forecast5, test, pca.style_names, 5)
                    oos_ic20 = self._mean_forecast_ic(forecast20, test, pca.style_names, 20)
                    for horizon, fitted in ((5, model5), (20, model20)):
                        importance = fitted.feature_importance_frame()
                        importance["quarter"], importance["horizon"] = str(quarter), horizon
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
                        aligned["quarter"], aligned["horizon"] = str(quarter), horizon
                        style_forecast_audit.append(aligned)
                    dynamic = self._dynamic_horizon_weights(timing, test, pca.style_names,
                        model5.validation_rank_ic, model20.validation_rank_ic,
                        available_forecast_ic)
                    forecast_ic_history.append(quarter_forecast_ic)
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
                    residual_model = ResidualRidge(pca.loadings, self.config.residual_alpha).fit(
                        residual_train, features, label="residual_label", sample_weight="sample_weight")
                    if residual_model.max_label_exit_date_ is not None and residual_model.max_label_exit_date_ >= cutoff:
                        raise AssertionError("残差模型训练包含尚未实现的未来标签")
                    quarter_factors = processed[processed.factor_date.isin(test.decision_date)]
                    residual_score = self._residual_score(residual_model, quarter_factors, features)
                    score = score.rename(columns={"prediction": "style_timing_score"}).merge(
                        residual_score, on=["factor_date", "symbol"], how="left", validate="one_to_one")
                    # 残差模型预测的是风格无法解释的收益增量，因此与风格分数相加。
                    # NaN风格分数仍为非买入候选，不由残差模型强行补仓。
                    score["prediction"] = self._combine_style_residual(
                        score.style_timing_score, score.residual_prediction_score,
                        self.config.residual_weight)
                    score = score.merge(
                        votes[["factor_date", "symbol", "buy_votes", "sell_votes", "buy_eligible"]],
                        on=["factor_date", "symbol"], how="left", validate="one_to_one")
                    risk = dynamic[["decision_date", "risk_off"]].rename(
                        columns={"decision_date": "factor_date"})
                    score = score.merge(risk, on="factor_date", how="left", validate="many_to_one")
                    coefficient = residual_model.coefficient_frame(features)
                    coefficient["quarter"] = str(quarter)
                    coefficient["train_end"] = residual_model.max_label_exit_date_
                    residual_coefficients.append(coefficient)
                    model_scores.append(score); model_votes.append(votes)
                    dynamic["model"] = output_name; model_weights.append(dynamic)
                    training_rows.append({"model": output_name, "quarter": str(quarter),
                        "train_end": str(max(train5.label_exit_date_5.max(), train20.label_exit_date_20.max())),
                        "train_start_5": str(train5.decision_date.min()), "train_end_5": str(train5.label_exit_date_5.max()),
                        "train_start_20": str(train20.decision_date.min()), "train_end_20": str(train20.label_exit_date_20.max()),
                        "train_samples_5": len(train5), "train_samples_20": len(train20),
                        "train_samples": len(train5) + len(train20), "test_days": len(test),
                        "residual_train_samples": residual_model.train_rows_,
                        "residual_train_end": str(residual_model.max_label_exit_date_),
                        "residual_alpha": self.config.residual_alpha,
                        "residual_weight": self.config.residual_weight,
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
            for quarter, context in contexts.items():
                pca, timing, test, cutoff = (context["pca"], context["timing"],
                                              context["test"], context["cutoff"])
                train = timing[(timing.label_exit_date < cutoff) & timing.best_style.notna()].copy()
                age = (cutoff - train.decision_date).dt.days / 365.25 * 252
                train["sample_weight"] = np.exp(-np.log(2) * age / STYLE_TIME_DECAY_HALFLIFE) * (1 + train.winner_margin.clip(lower=0))
                timer = StyleTimer(model_name, pca.style_names).fit(train)
                weight = timer.predict_weights(test)
                model_weights.append(weight)
                training_rows.append({"model": output_name, "quarter": str(quarter), "train_end": str(train.label_exit_date.max()),
                                      "train_samples": len(train), "test_days": len(test)})
                quarter_exposure = context["exposure"][
                    context["exposure"].factor_date.isin(test.decision_date)]
                model_scores.append(self._stock_scores(quarter_exposure, weight, pca.style_names))
            weights = pd.concat(model_weights, ignore_index=True); weights["model"] = output_name
            weight_frames.append(weights)
            score = pd.concat(model_scores, ignore_index=True)
            score = score.merge(labels[["factor_date", "symbol", "label"]], on=["factor_date", "symbol"], how="left")
            score["model"] = output_name; predictions.append(score)
        prediction = pd.concat(predictions, ignore_index=True)
        weights = pd.concat(weight_frames, ignore_index=True)
        vote_audit = pd.concat(vote_frames, ignore_index=True) if vote_frames else pd.DataFrame()
        daily = pd.concat([daily_ic(g).assign(model=m) for m, g in prediction.groupby("model")], ignore_index=True)
        component_daily_frame = self._component_daily_metrics(prediction)
        summary = layered_metrics(prediction)
        training_frame = pd.DataFrame(training_rows)
        if not training_frame.empty:
            quarterly = daily.assign(quarter=daily.factor_date.dt.to_period("Q").astype(str)).groupby(
                ["model", "quarter"], observed=True).rank_ic.agg(["mean", "std"]).reset_index()
            quarterly["quarter_icir"] = quarterly["mean"] / quarterly["std"].replace(0, np.nan)
            training_frame = training_frame.merge(
                quarterly.rename(columns={"mean": "quarter_rank_ic"})[["model", "quarter", "quarter_rank_ic", "quarter_icir"]],
                on=["model", "quarter"], how="left")
            if "train_rank_ic" not in training_frame:
                training_frame["train_rank_ic"] = np.nan
        # 完整逐股票预测与其他模型一致使用Parquet；避免重复写百万行CSV。
        prediction.to_parquet(self.logger.root / "predictions/all_predictions.parquet", index=False)
        for model_name, group in prediction.groupby("model"):
            target = self.logger.root / "predictions/model_prediction_files" / f"{model_name}_predictions.parquet"
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
