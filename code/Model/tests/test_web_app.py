from imports import *
import pytest
import web_app
from web_app import (TrainingJob, json_safe, optimizer_info, save_factor_upload,
                     save_industry_upload, validate_period_config, validate_style_config)
from env import MODEL_ROOT, SUPPORTED_POOLS


def test_web_job_rejects_empty_model_and_pool():
    job = TrainingJob()
    with pytest.raises(ValueError):
        job.start({"models": [], "pools": ["A500"]})
    with pytest.raises(ValueError):
        job.start({"models": ["ridge"], "pools": []})


def test_all_market_options_are_supported_and_rendered():
    assert {"ALL_A500", "ALL_ZZ1000", "ALL_MARKET"}.issubset(SUPPORTED_POOLS)
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert 'value="ALL_A500"' in html
    assert 'value="ALL_ZZ1000"' in html
    assert 'value="ALL_MARKET"' in html
    assert 'id="cfgXgbDepth"' in html
    assert 'id="factorSource"' in html
    assert 'value="daily_alpha"' in html
    assert 'value="monthly_calc"' in html
    assert "factor_source:factorSource" in html
    assert "factor_source:factorSource" in html and "modelFactorSource" in html


def test_web_job_rejects_unknown_builtin_factor_source():
    job = TrainingJob()
    with pytest.raises(ValueError, match="内置因子文件"):
        job.start({"models": ["ridge"], "pools": ["A500"],
                   "factor_source": "unknown"})


def test_web_job_keeps_rebalance_frequency_in_worker_config(monkeypatch):
    job = TrainingJob()
    captured = {}
    monkeypatch.setattr(job, "_run", lambda config: captured.update(config))
    job.start({"models": ["ridge"], "pools": ["A500"],
               "rebalance_frequency": "weekly"})
    for _ in range(20):
        if captured:
            break
        time.sleep(0.01)
    assert captured["rebalance_frequency"] == "weekly"


def test_fixed_rolling_window_derives_initial_cutoff_from_first_test_year(monkeypatch):
    job = TrainingJob()
    captured = {}
    monkeypatch.setattr(job, "_run", lambda config: captured.update(config))
    job.start({"models": ["ridge"], "pools": ["A500"],
               "factor_source": "monthly_calc", "selected_factors": ["calc_size"],
               "window_mode": "fixed", "train_start": "2010-01-01",
               "train_end": "2024-06-30", "test_start": "2025-06-01",
               "test_end": "2026-08-31"})
    for _ in range(20):
        if captured:
            break
        time.sleep(0.01)
    assert captured["train_end"] == "2024-12-31"


@pytest.mark.parametrize("mode, expected_start", [
    ("expanding", "2009-04-30"),
    ("fixed", "2020-01-02"),
])
def test_model_backtest_range_derives_internal_training_history(
        monkeypatch, mode, expected_start):
    job = TrainingJob()
    captured = {}
    monkeypatch.setattr(job, "_run", lambda config: captured.update(config))
    job.start({"models": ["ridge"], "pools": ["A500"],
               "factor_source": "monthly_calc", "selected_factors": ["calc_size"],
               "window_mode": mode, "train_years": 4,
               "backtest_start": "2024-01-01", "backtest_end": "2024-12-31"})
    for _ in range(30):
        if captured:
            break
        time.sleep(.01)
    assert captured["train_start"] == expected_start
    assert captured["train_end"] == "2023-12-31"
    assert captured["test_start"] == "2024-01-01"
    assert captured["test_end"] == "2024-12-31"


def test_old_frontend_request_before_daily_coverage_uses_monthly_factor(monkeypatch):
    job = TrainingJob()
    captured = {}
    monkeypatch.setattr(job, "_run", lambda config: captured.update(config))
    job.start({"models": ["ridge"], "pools": ["ALL_MARKET"],
               "train_start": "2018-01-01", "train_end": "2019-12-31",
               "test_start": "2020-01-01", "test_end": "2020-12-31"})
    for _ in range(20):
        if captured:
            break
        time.sleep(0.01)
    assert captured["factor_source"] == "monthly_calc"
    assert captured["factor_path"].endswith("data_in_sample_monthly.parquet")


def test_legacy_request_without_factor_source_defaults_to_monthly_library(monkeypatch):
    job = TrainingJob()
    captured = {}
    monkeypatch.setattr(job, "_run", lambda config: captured.update(config))
    job.start({"models": ["ridge"], "pools": ["A500"],
               "train_start": "2020-01-01", "train_end": "2023-12-31",
               "test_start": "2024-01-01", "test_end": "2024-12-31"})
    for _ in range(20):
        if captured:
            break
        time.sleep(0.01)
    assert captured["factor_source"] == "monthly_calc"
    assert captured["factor_frequency"] == "monthly"


def test_style_xgb_depth_is_validated_and_preserved():
    assert validate_style_config({"xgb_max_depth": 4})["xgb_max_depth"] == 4
    assert validate_style_config({"rebalance_frequency": "weekly"})[
        "rebalance_frequency"] == "weekly"
    assert validate_style_config({"rebalance_frequency": "quarterly"})[
        "rebalance_frequency"] == "quarterly"
    with pytest.raises(ValueError, match="最大深度"):
        validate_style_config({"xgb_max_depth": 11})
    with pytest.raises(ValueError, match="调仓频率"):
        validate_style_config({"rebalance_frequency": "yearly"})


def test_factor_upload_validates_and_normalizes_long_table(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "FACTOR_UPLOAD_DIR", tmp_path)
    result = save_factor_upload(
        "signals.csv", b"date,ticker,value_factor\n20240102,1,0.5\n20240103,2,0.7\n")
    assert result["features"] == ["value_factor"]
    saved = pd.read_parquet(tmp_path / "signals.parquet")
    assert saved["ticker"].tolist() == ["000001", "000002"]
    assert set(saved.columns) == {"date", "ticker", "value_factor"}


def test_period_config_rejects_overlapping_train_and_test():
    with pytest.raises(ValueError, match="训练区间"):
        validate_period_config({"train_end": "2024-01-01", "test_start": "2023-12-31"})


def test_optimizer_selector_and_industry_upload_are_rendered():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert 'id="optimizer"' in html
    assert 'id="turnoverOptimizer"' in html
    assert 'value="lazy_turnover"' in html
    assert 'id="drawdownOptimizer"' in html
    assert 'value="quadratic_drawdown"' in html
    assert 'id="sellConfirmations"' in html
    assert 'id="maxTurnoverRatio"' in html
    assert 'id="maxDrawdownLimit"' in html
    assert 'value="industry_neutral"' in html
    assert 'id="industryFile"' in html
    assert "/api/industry-upload" in html
    assert 'id="optTurnover" type="checkbox"' in html
    assert 'id="optDrawdown" type="checkbox"' in html
    assert 'id="optIndustry" type="checkbox"' in html
    assert "turnover_limit" in html
    assert "backtest_stage:stage" in html


def test_model_workflow_is_ordered_and_scrollable():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert all(label in html for label in
               ("STEP 01", "STEP 02", "STEP 03", "STEP 04",
                "纯模型", "模型 + 手续费", "模型 + 优化器"))
    assert ".workspace{min-height:0;overflow-y:auto" in html


def test_backtest_stage_controls_costs_and_optimizers(monkeypatch):
    def captured(raw):
        job = TrainingJob()
        value = {}
        monkeypatch.setattr(job, "_run", lambda config: value.update(config))
        job.start(raw)
        for _ in range(30):
            if value:
                break
            time.sleep(.01)
        return value

    base = {"models": ["ridge"], "pools": ["A500"],
            "factor_source": "daily_alpha",
            "train_start": "2020-01-02", "train_end": "2023-12-31",
            "test_start": "2024-01-01", "test_end": "2024-12-31"}
    pure = captured({**base, "backtest_stage": "pure",
                     "optimizer": "industry_neutral",
                     "turnover_optimizer": "turnover_limit"})
    assert pure["include_costs"] is False
    assert pure["optimizer"] == pure["turnover_optimizer"] == "none"

    cost = captured({**base, "backtest_stage": "cost"})
    assert cost["include_costs"] is True
    assert cost["optimizer"] == cost["drawdown_optimizer"] == "none"

    optimized = captured({**base, "backtest_stage": "optimized",
                          "turnover_optimizer": "turnover_limit",
                          "drawdown_optimizer": "quadratic_drawdown"})
    assert optimized["include_costs"] is True
    assert optimized["turnover_optimizer"] == "turnover_limit"
    assert optimized["drawdown_optimizer"] == "quadratic_drawdown"


def test_web_job_rejects_unknown_optimizer():
    job = TrainingJob()
    with pytest.raises(ValueError, match="合法优化器"):
        job.start({"models": ["ridge"], "pools": ["A500"], "optimizer": "unknown"})
    with pytest.raises(ValueError, match="换手率优化器"):
        job.start({"models": ["ridge"], "pools": ["A500"],
                   "turnover_optimizer": "unknown"})
    with pytest.raises(ValueError, match="最大回撤优化器"):
        job.start({"models": ["ridge"], "pools": ["A500"],
                   "drawdown_optimizer": "unknown"})


def test_model_frontend_uses_multi_select_dropdown():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert 'id="modelDropdown" class="multi-select"' in html
    assert html.count('name="model"') == 5
    assert 'models=selected(\'model\')' in html
    assert 'models,pools:selected(\'pool\')' in html
    assert 'id="factorDrop"' in html
    assert "/api/factor-upload" in html
    assert 'id="trainStart"' in html
    assert 'id="testEnd"' in html


def test_model_timing_controls_live_only_in_model_stage():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    factor_stage = html.split('data-stage-pane="1"', 1)[1].split(
        'data-stage-pane="2"', 1)[0]
    model_stage = html.split('data-stage-pane="2"', 1)[1].split(
        'data-stage-pane="3"', 1)[0]
    assert 'id="backtestStart"' not in factor_stage
    assert 'id="backtestEnd"' not in factor_stage
    assert 'id="windowMode"' not in factor_stage
    assert all(control in model_stage for control in (
        'id="windowMode"', 'id="backtestStart"', 'id="backtestEnd"',
        'id="rollingYearsField"'))
    assert "自动初始训练截止" not in html
    assert "backtest_start:backtestStart" in html
    assert "backtest_end:backtestEnd" in html


def test_factor_catalog_pickers_are_rendered_for_analysis_and_models():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert 'id="factorBacktestSource"' in html
    assert 'id="factorFactorRows"' in html
    assert 'id="modelFactorRows"' in html
    assert "序号" in html and "表达式 / 定义" in html
    assert "selectFactorRows('factor',true)" in html
    assert "selectFactorRows('model',true)" in html
    assert "selected_factors:selectedFactors" in html
    assert 'id="factorNames"' not in html


def test_web_job_validates_and_preserves_selected_factors(monkeypatch):
    job = TrainingJob()
    captured = {}
    monkeypatch.setattr(job, "_run", lambda config: captured.update(config))
    job.start({"models": ["ridge"], "pools": ["A500"],
               "factor_source": "daily_alpha", "selected_factors": ["alpha_2", "alpha_0"],
               "train_start": "2020-01-02", "train_end": "2023-12-31",
               "test_start": "2024-01-01", "test_end": "2024-12-31"})
    for _ in range(30):
        if captured:
            break
        time.sleep(.01)
    assert captured["selected_factors"] == ["alpha_2", "alpha_0"]

    with pytest.raises(ValueError, match="至少选择一个模型因子"):
        TrainingJob().start({"models": ["ridge"], "pools": ["A500"],
                             "factor_source": "daily_alpha", "selected_factors": []})
    with pytest.raises(ValueError, match="不在当前因子文件"):
        TrainingJob().start({"models": ["ridge"], "pools": ["A500"],
                             "factor_source": "daily_alpha",
                             "selected_factors": ["not_a_factor"]})


def test_worker_command_passes_selected_factors_to_both_entrypoints():
    source = (MODEL_ROOT / "web_app.py").read_text(encoding="utf-8")
    features_position = source.index('command.extend(["--features", *config["selected_factors"]])')
    model_branch_position = source.index('if model != "style_rotation":', features_position)
    assert features_position < model_branch_position


def test_lazy_turnover_is_supported_only_for_direct_pca_xgboost():
    job = TrainingJob()
    with pytest.raises(ValueError):
        job.start({"models": ["ridge"], "pools": ["A500"],
                   "turnover_optimizer": "lazy_turnover"})


def test_industry_upload_is_validated_and_saved_atomically(tmp_path, monkeypatch):
    csv_path, parquet_path = tmp_path / "industry.csv", tmp_path / "industry.parquet"
    monkeypatch.setattr(web_app, "DATA_DIR", tmp_path)
    monkeypatch.setattr(web_app, "INDUSTRY_DATA_CANDIDATES", [parquet_path, csv_path])
    result = save_industry_upload(
        "classification.csv", b"ticker,industry\n000001,bank\n000002,tech\n")
    status = optimizer_info()
    assert result["symbols"] == 2
    assert csv_path.exists()
    assert status["industry_ready"] is True
    assert status["industry_symbols"] == 2
    assert not list(tmp_path.glob("industry.upload-*"))


def test_style_parameter_modal_explains_every_parameter():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert html.count('class="param-help"') == 18
    assert "pca_alignment_threshold:.80" in html
    assert "pca_explained_threshold:.85" in html
    assert "confidence_rankic_window:60" in html
    assert 'id="cfgHoldingCount"' in html and 'id="cfgRotationQuantile"' in html
    assert 'id="cfgRebalanceFrequency"' in html
    assert 'id="cfgPcaFitMode"' in html
    assert all(f'value="{value}"' in html for value in
               ("daily", "alternate", "weekly", "monthly"))
    assert "['cfgBuy','cfgVoteQuantile','cfgResidualWeight','cfgResidualAlpha','cfgIcirWindow']" in html
    assert "PCA-XGBoost 参数" in html


def test_model_output_visualizes_pca_metrics_and_score_layers():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert "absolute_calmar" in html
    assert "absolute_profit_loss_ratio" in html
    assert "absolute_underwater_ratio" in html
    assert "按模型得分分层净值" in html
    assert "q10_q1" in html


def test_web_style_entry_runs_direct_stock_pca_xgboost():
    source = (MODEL_ROOT / "web_app.py").read_text(encoding="utf-8")
    assert 'command.extend(["--style-models", "pca_stock_xgboost"])' in source
    assert 'command.extend(["--style-models", "xgboost_dual_horizon"])' not in source


def test_frontend_displays_pool_icir_metrics():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert "票池ICIR" in html
    assert "票池年化ICIR" not in html


def test_style_frontend_compares_same_horizon_cross_style_ic():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert "5日 训练跨风格IC" in html
    assert "5日 固定20日隔离验证IC（单点）" not in html
    assert "5日 样本外跨风格IC" in html
    assert "20日 训练跨风格IC" in html
    assert "20日 固定20日隔离验证IC（单点）" not in html
    assert "20日 样本外跨风格IC" in html
    assert "pca_style_ic" in html
    assert "PCA '+style+' 股票RankIC" in html


def test_web_job_starts_idle_and_has_serial_state():
    snapshot = TrainingJob().snapshot()
    assert snapshot["state"] == "idle"
    assert snapshot["completed_models"] == []


def test_delete_run_removes_only_the_requested_registered_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "OUTPUT_ROOT", tmp_path)
    first = tmp_path / "run_20260915_120000_xgboost"
    second = tmp_path / "run_20260915_120001_xgboost"
    first.mkdir(); second.mkdir()
    (first / "result.txt").write_text("first", encoding="utf-8")
    (second / "result.txt").write_text("second", encoding="utf-8")
    job = TrainingJob()
    job.run_dirs = {first.name: str(first), second.name: str(second)}
    job.run_models = {first.name: "xgboost", second.name: "xgboost"}

    job.delete_run(first.name)

    assert not first.exists()
    assert second.exists()
    assert second.name in job.snapshot()["model_results"]


def test_same_model_runs_persist_as_distinct_results(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "OUTPUT_ROOT", tmp_path)
    run_ids = ["run_20260915_120000_xgboost", "run_20260915_120001_xgboost"]
    for run_id in run_ids:
        (tmp_path / run_id).mkdir()
    job = TrainingJob()
    job.run_dirs = {run_id: str(tmp_path / run_id) for run_id in run_ids}
    job.run_models = {run_id: "xgboost" for run_id in run_ids}
    job._save_runs()

    restored = TrainingJob()
    restored._load_runs()

    assert set(restored.snapshot()["model_results"]) == set(run_ids)


def test_run_registry_is_rebuilt_from_completed_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "OUTPUT_ROOT", tmp_path)
    run_id = "run_20260915_120000_123456_linear_regression"
    run = tmp_path / run_id
    (run / "config").mkdir(parents=True)
    (run / "config/run_manifest.json").write_text(
        json.dumps({"models": ["linear_regression"]}), encoding="utf-8")

    restored = TrainingJob()
    restored._load_runs()

    assert restored.run_models[run_id] == "linear_regression"
    assert json.loads((tmp_path / "run_registry.json").read_text(encoding="utf-8"))[0][
        "run_id"] == run_id


def test_completed_model_panels_keep_backtest_metrics(tmp_path):
    run = tmp_path / "run_model"
    (run / "backtest/results").mkdir(parents=True)
    pd.DataFrame([{"model": "ridge", "pool": "A500", "absolute_sharpe": 1.2,
                   "absolute_max_drawdown": -.08}]).to_csv(run / "backtest/results/summary.csv", index=False)
    job = TrainingJob()
    job.run_dirs = {"ridge": str(run), "elasticnet": str(tmp_path / "next")}
    snapshot = job.snapshot()
    metrics = snapshot["model_results"]["ridge"]["backtest_metrics"][0]
    assert metrics["absolute_sharpe"] == 1.2
    assert metrics["absolute_max_drawdown"] == -.08


def test_curves_are_keyed_by_the_same_pool_as_metrics(tmp_path):
    run = tmp_path / "run_model"
    results = run / "backtest/results"
    results.mkdir(parents=True)
    pd.DataFrame([{"model": "ridge", "pool": "A500", "absolute_sharpe": -1.0},
                  {"model": "ridge", "pool": "ZZ1000", "absolute_sharpe": 1.0}]).to_csv(results / "summary.csv", index=False)
    for pool, end in [("A500", .8), ("ZZ1000", 1.1)]:
        (results / f"ridge_{pool}_live.json").write_text(json.dumps(
            {"model": "ridge", "pool": pool, "curve": [{"date": "2024-01-02", "strategy": end}]}), encoding="utf-8")
    artifacts = TrainingJob()._artifacts(run)
    assert artifacts["backtests"]["A500"]["curve"][0]["strategy"] == .8
    assert artifacts["backtests"]["ZZ1000"]["curve"][0]["strategy"] == 1.1


def test_json_safe_replaces_non_finite_numbers():
    payload = json_safe({"strategy": np.nan, "values": [np.inf, -np.inf, np.float64(1.2)]})
    assert payload == {"strategy": None, "values": [None, None, 1.2]}
    assert "NaN" not in json.dumps(payload, allow_nan=False)


def test_stop_finishes_recovered_stopping_state():
    job = TrainingJob()
    job.state = "stopping"
    job.process = None
    job.process_pid = None
    job.stop()
    assert job.state == "stopped"
    assert "任务已终止" in job.logs[-1]


def test_global_stop_button_is_available_outside_model_stage():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    header = html.split("<header>", 1)[1].split("</header>", 1)[0]
    assert 'id="stopTask"' in header
    assert 'onclick="stopJob()"' in header
    assert "全部子进程" in header


def test_process_tree_termination_uses_taskkill_on_windows(monkeypatch):
    calls = []
    monkeypatch.setattr(web_app.os, "name", "nt")
    monkeypatch.setattr(web_app.subprocess, "run",
                        lambda command, **kwargs: calls.append(command))
    TrainingJob._terminate_process_tree(None, 12345)
    assert calls == [["taskkill", "/PID", "12345", "/T", "/F"]]
