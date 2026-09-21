from imports import *
from env import REPORT_MAX_TABLE_ROWS

# Plotly会在创建任意Figure时加载已弃用但本项目未使用的scattermapbox验证器。
warnings.filterwarnings("ignore", message="(?s).*scattermapbox.*", category=DeprecationWarning)


class ReportGenerator:
    """生成无需服务器、无需CDN、可直接双击打开的单文件报告。"""

    @staticmethod
    def _table(title: str, frame: pd.DataFrame, full_path: str = "") -> str:
        note = f"<p>完整文件：{html.escape(full_path)}</p>" if full_path else ""
        return (f"<details><summary>{html.escape(title)}（展示前{REPORT_MAX_TABLE_ROWS}行）</summary>"
                f"{note}{frame.head(REPORT_MAX_TABLE_ROWS).to_html(index=False, border=0)}</details>")

    @staticmethod
    def _diagnosis(summaries: pd.DataFrame, trials: pd.DataFrame, training: pd.DataFrame,
                   backtests: pd.DataFrame) -> list[str]:
        messages = []
        if summaries.empty:
            return ["样本外失效：没有成功模型。"]
        if trials is not None and not trials.empty and "value" in trials:
            ordered = trials.sort_values("number")
            best = ordered.value.cummax()
            recent_gain = best.iloc[-1] - best.iloc[max(0, len(best) - 20)]
            messages.append("持续改善：后20个Trial仍有提升。" if recent_gain > 1e-4 else "已经收敛：后20个Trial改善有限。")
            if "user_attrs_train_rank_ic" in ordered and ordered.iloc[-1].user_attrs_train_rank_ic - ordered.iloc[-1].value > .05:
                messages.append("可能过拟合：训练RankIC明显高于验证RankIC。")
        for model, group in training.groupby("model") if not training.empty else []:
            values = group.sort_values("quarter").quarter_rank_ic.dropna()
            if len(values) >= 3 and values.is_monotonic_decreasing:
                messages.append(f"{model} 样本外RankIC逐季下降。")
        if (summaries.mean_rank_ic <= 0).any():
            messages.append("样本外失效：至少一个模型全年Mean RankIC不为正。")
        if not backtests.empty and "turnover" in backtests:
            messages.append("回测收益需与换手率和交易费联合判断，诊断不会反向修改参数。")
        return messages or ["结果不稳定：当前证据不足以判断持续改善。"]

    def generate(self, path: Path, run_id: str, config: dict, daily_metrics: pd.DataFrame,
                 summaries: pd.DataFrame, training_records: pd.DataFrame, failures=None,
                 trials=None, backtests=None, backtest_curves=None, details=None,
                 parameter_importance=None, feature_importance=None, manifest=None,
                 pca_style_metrics=None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trials = pd.DataFrame() if trials is None else trials
        backtests = pd.DataFrame() if backtests is None else backtests
        backtest_curves = pd.DataFrame() if backtest_curves is None else backtest_curves
        pca_style_metrics = (pd.DataFrame() if pca_style_metrics is None
                             else pca_style_metrics)
        details = details or {}

        daily_fig = go.Figure()
        cumulative_fig = go.Figure()
        quarterly_fig = go.Figure()
        daily_groups = (["model", "pool"] if "pool" in daily_metrics.columns else ["model"])
        for keys, group in daily_metrics.sort_values("factor_date").groupby(daily_groups) if not daily_metrics.empty else []:
            label = "-".join(keys) if isinstance(keys, tuple) else keys
            daily_fig.add_scatter(x=group.factor_date, y=group.rank_ic, name=label)
            cumulative_fig.add_scatter(x=group.factor_date, y=group.rank_ic.fillna(0).cumsum(), name=label)
            quarter = group.assign(quarter=group.factor_date.dt.to_period("Q").astype(str)).groupby("quarter").rank_ic.mean()
            quarterly_fig.add_scatter(x=quarter.index, y=quarter.values, name=label, mode="lines+markers")
        daily_fig.update_layout(title="每日RankIC", template="plotly_white")
        cumulative_fig.update_layout(title="每日RankIC累计和", template="plotly_white")
        quarterly_fig.update_layout(title="最终股票分数季度Mean RankIC", template="plotly_white")

        style_generalization_fig = go.Figure()
        style_columns = {"train_rank_ic_5", "oos_style_rank_ic_5",
                         "train_rank_ic_20", "oos_style_rank_ic_20"}
        if not training_records.empty and style_columns.issubset(training_records.columns):
            group_columns = ["model", "pool"] if "pool" in training_records.columns else ["model"]
            ordered = training_records.sort_values([*group_columns, "quarter"])
            for keys, group in ordered.groupby(group_columns):
                label = "-".join(keys) if isinstance(keys, tuple) else keys
                for horizon in (5, 20):
                    for column, scope in ((f"train_rank_ic_{horizon}", "训练"),
                                          (f"oos_style_rank_ic_{horizon}", "样本外")):
                        style_generalization_fig.add_scatter(
                            x=group.quarter, y=group[column], name=f"{label} {horizon}日{scope}",
                            mode="lines+markers")
        if not pca_style_metrics.empty:
            group_columns = ["pool", "style"] if "pool" in pca_style_metrics.columns else ["style"]
            for keys, group in pca_style_metrics.sort_values("quarter").groupby(group_columns):
                label = "-".join(keys) if isinstance(keys, tuple) else keys
                style_generalization_fig.add_scatter(
                    x=group.quarter, y=group.mean_rank_ic,
                    name=f"PCA {label} 股票RankIC", mode="lines+markers")
        style_generalization_fig.update_layout(
            title="风格模型拟合/调参与PCA成分股票RankIC",
            template="plotly_white")

        trial_fig = go.Figure()
        if not trials.empty and {"number", "value"}.issubset(trials.columns):
            trial_fig.add_scatter(x=trials.number, y=trials.value, name="验证Mean RankIC", mode="markers")
            trial_fig.add_scatter(x=trials.number, y=trials.value.cummax(), name="Best-so-far")
        trial_fig.update_layout(title="Optuna优化过程", template="plotly_white")

        nav_fig, excess_fig, drawdown_fig, quarter_return_fig = (go.Figure() for _ in range(4))
        if not backtest_curves.empty:
            for (model, pool), group in backtest_curves.groupby(["model", "pool"]):
                label = f"{model}-{pool}"
                nav_fig.add_scatter(x=group.date, y=group.strategy, name=f"{label} 策略")
                nav_fig.add_scatter(x=group.date, y=group.benchmark, name=f"{label} 基准", line={"dash": "dot"})
                excess_fig.add_scatter(x=group.date, y=group.excess, name=label)
                drawdown = group.strategy / group.strategy.cummax() - 1
                drawdown_fig.add_scatter(x=group.date, y=drawdown, name=label, fill="tozeroy")
                returns = group.set_index("date").strategy.resample("QE").last().pct_change()
                quarter_return_fig.add_bar(x=returns.index.astype(str), y=returns.values, name=label)
        for fig, title in [(nav_fig, "策略与基准净值"), (excess_fig, "相对净值"),
                           (drawdown_fig, "策略回撤"), (quarter_return_fig, "季度收益")]:
            fig.update_layout(title=title, template="plotly_white")

        overview = {"run_id": run_id, "train_range": f"{config.get('train_start')} ~ {config.get('initial_train_end')}",
            "test_range": f"{config.get('test_start')} ~ {config.get('test_end')}", "models": config.get("models"),
            "pools": config.get("pools"), "optuna_trials": config.get("optuna_trials"),
            "duration_seconds": (manifest or {}).get("duration"), "data_audit_passed": (manifest or {}).get("data_audit_passed"),
            "xgboost_best_params": (manifest or {}).get("annual_parameters", {}).get("xgboost")}
        cards = "".join(f"<div class='card'><b>{html.escape(str(k))}</b><br>{html.escape(str(v))}</div>" for k, v in overview.items())
        diagnosis = "".join(f"<li>{html.escape(x)}</li>" for x in self._diagnosis(summaries, trials, training_records, backtests))
        figures = [daily_fig, cumulative_fig, quarterly_fig, style_generalization_fig,
                   trial_fig, nav_fig, excess_fig, drawdown_fig, quarter_return_fig]
        with warnings.catch_warnings():
            # Plotly当前版本会为未使用的scattermapbox验证器发弃用警告，和本报告图形无关。
            warnings.filterwarnings("ignore", message="(?s).*scattermapbox.*", category=DeprecationWarning)
            figure_html = "".join(fig.to_html(full_html=False, include_plotlyjs=True if index == 0 else False)
                                  for index, fig in enumerate(figures))
        tables = [self._table("模型汇总", summaries, "metrics/model_summary.csv"),
                  self._table("季度训练记录", training_records, "metrics/training_records.csv"),
                  self._table("每日RankIC", daily_metrics, "metrics/daily_rankic.csv"),
                  self._table("Optuna Trial", trials, "optuna/<year>_trials.csv"),
                  self._table("Optuna参数重要性", pd.DataFrame() if parameter_importance is None else parameter_importance, "optuna/<year>_parameter_importance.csv"),
                  self._table("模型特征重要性", pd.DataFrame() if feature_importance is None else feature_importance, "models/<model>/<quarter>/feature_importance.csv"),
                  self._table("PCA每个风格RankIC", pca_style_metrics, "metrics/pca_style_rankic.csv"),
                  self._table("回测汇总", backtests, "backtest/results/summary.csv")]
        labels = {"account": "账户每日记录", "orders": "成交记录", "trades": "平仓交易记录",
                  "holdings": "每日收盘持仓", "adjustments": "复权因子调整记录"}
        tables.extend(self._table(labels[key], value, f"backtest/{key}/") for key, value in details.items() if key in labels)
        tables.append(self._table("失败任务日志", pd.DataFrame(failures or []), "logs/failed_tasks.csv"))
        document = f"""<!doctype html><html><head><meta charset='utf-8'><title>滚动机器学习报告</title>
<style>body{{font-family:Arial,'Microsoft YaHei';margin:28px;background:#f5f7fb;color:#172033}}h1,h2{{color:#172033}}.cards{{display:flex;gap:12px;flex-wrap:wrap}}.card,details,.diagnosis{{background:white;padding:15px;border-radius:9px;margin:10px 0;box-shadow:0 1px 4px #ccd2dc}}table{{border-collapse:collapse;width:100%;font-size:12px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}summary{{font-weight:bold;cursor:pointer}}</style></head><body>
<h1>滚动机器学习样本外报告</h1><h2>运行概览</h2><div class='cards'>{cards}</div>
<div class='diagnosis'><h2>客观诊断</h2><ul>{diagnosis}</ul></div>{figure_html}<h2>明细表</h2>{''.join(tables)}</body></html>"""
        path.write_text(document, encoding="utf-8")
        return path
