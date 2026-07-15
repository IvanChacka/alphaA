from imports import *
import pytest
import web_app
from web_app import TrainingJob, json_safe, optimizer_info, save_industry_upload
from env import MODEL_ROOT, SUPPORTED_POOLS


def test_web_job_rejects_empty_model_and_pool():
    job = TrainingJob()
    with pytest.raises(ValueError):
        job.start({"models": [], "pools": ["A500"]})
    with pytest.raises(ValueError):
        job.start({"models": ["ridge"], "pools": []})


def test_all_market_options_are_supported_and_rendered():
    assert {"ALL_A500", "ALL_ZZ1000"}.issubset(SUPPORTED_POOLS)
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert 'value="ALL_A500"' in html
    assert 'value="ALL_ZZ1000"' in html


def test_optimizer_selector_and_industry_upload_are_rendered():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert 'id="optimizer"' in html
    assert 'value="industry_neutral"' in html
    assert 'id="industryFile"' in html
    assert "/api/industry-upload" in html
    assert "optimizer:$('optimizer').value" in html


def test_web_job_rejects_unknown_optimizer():
    job = TrainingJob()
    with pytest.raises(ValueError, match="合法优化器"):
        job.start({"models": ["ridge"], "pools": ["A500"], "optimizer": "unknown"})


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
    assert html.count('class="param-help"') == 8
    assert "L2正则强度而不是IC目标" in html
    assert "0.90表示每个风格仅最强10%看多" in html
    assert 'id="cfgResidualAlpha" type="number" min="0.000001" step="any"' in html


def test_frontend_displays_pool_icir_metrics():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert "票池ICIR" in html
    assert "票池年化ICIR" not in html


def test_style_frontend_compares_same_horizon_cross_style_ic():
    html = (MODEL_ROOT / "web/index.html").read_text(encoding="utf-8")
    assert "5日 训练跨风格IC" in html
    assert "5日 固定20日隔离验证IC（单点）" in html
    assert "5日 样本外跨风格IC" in html
    assert "20日 训练跨风格IC" in html
    assert "20日 固定20日隔离验证IC（单点）" in html
    assert "20日 样本外跨风格IC" in html
    assert "tr.map(x=>x.validation_rank_ic_5)" not in html
    assert "tr.map(x=>x.validation_rank_ic_20)" not in html


def test_web_job_starts_idle_and_has_serial_state():
    snapshot = TrainingJob().snapshot()
    assert snapshot["state"] == "idle"
    assert snapshot["completed_models"] == []


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
