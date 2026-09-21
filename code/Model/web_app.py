"""滚动模型实时控制台：顺序执行模型、展示进度并支持终止。"""
from imports import *
from email import policy
from email.parser import BytesParser
from urllib.parse import parse_qs
from env import (DATA_DIR, DRAWDOWN_LIMIT, DRAWDOWN_OPTIMIZER_NAMES,
                 FACTOR_SOURCES, FEATURE_EXCLUDED_COLUMNS, FEATURE_FORBIDDEN_PATTERNS,
                 INDUSTRY_DATA_CANDIDATES, MODEL_N_JOBS, MODEL_ROOT,
                 OPTIMIZER_NAMES, OUTPUT_ROOT, SUPPORTED_POOLS, TURNOVER_MAX_RATIO,
                 TURNOVER_OPTIMIZER_NAMES, TURNOVER_SELL_CONFIRMATIONS)
from optimizers import create_drawdown_optimizer, create_optimizer, create_turnover_optimizer
from optimizers.industry_neutral import IndustryNeutralOptimizer
from research_framework.data import LocalParquetSource
from research_framework.service import ResearchService, factor_columns


WEB_ROOT = MODEL_ROOT / "web"
FACTOR_UPLOAD_DIR = DATA_DIR / "uploaded_factors"
RESEARCH = ResearchService(FACTOR_SOURCES, DATA_DIR / "adj_close.pq" / "adj_close.pq" / "adj_close.parquet",
                           FACTOR_UPLOAD_DIR)


STYLE_CONFIG_DEFAULTS = {
    "pca_variance": 0.90,
    "pca_min_components": 5,
    "xgb_max_depth": 4,
    "pca_alignment_threshold": 0.80,
    "pca_explained_threshold": 0.85,
    "confidence_rankic_window": 60,
    "confidence_rankic_threshold": 0.0,
    "holding_count": 200,
    "rotation_quantile": 0.10,
    "rebalance_frequency": "daily",
    "pca_fit_mode": "rolling",
    "weight_mode": "score",
}


def optimizer_info() -> dict[str, Any]:
    existing = next((Path(path) for path in INDUSTRY_DATA_CANDIDATES if Path(path).exists()), None)
    payload = {"optimizers": OPTIMIZER_NAMES,
               "turnover_optimizers": TURNOVER_OPTIMIZER_NAMES,
               "drawdown_optimizers": DRAWDOWN_OPTIMIZER_NAMES,
               "industry_file": str(existing) if existing else "",
               "industry_ready": False, "industry_rows": 0, "industry_symbols": 0,
               "industry_dated": False, "error": ""}
    if existing:
        try:
            mapping = IndustryNeutralOptimizer.load_mapping(existing)
            payload.update({"industry_ready": True, "industry_rows": len(mapping),
                            "industry_symbols": int(mapping.symbol.nunique()),
                            "industry_dated": "date" in mapping})
        except Exception as exc:
            payload["error"] = str(exc)
    return payload


def save_industry_upload(filename: str, body: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError("行业分类文件仅支持.csv或.parquet")
    if not body:
        raise ValueError("上传的行业分类文件为空")
    if len(body) > 100 * 1024 * 1024:
        raise ValueError("行业分类文件不能超过100MB")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    target = DATA_DIR / f"industry{suffix}"
    temporary = DATA_DIR / f"industry.upload-{os.getpid()}-{threading.get_ident()}{suffix}"
    try:
        temporary.write_bytes(body)
        mapping = IndustryNeutralOptimizer.load_mapping(temporary)
        os.replace(temporary, target)
        for candidate in INDUSTRY_DATA_CANDIDATES:
            candidate = Path(candidate)
            if candidate != target and candidate.exists():
                candidate.unlink()
        return {"ok": True, "path": str(target), "rows": len(mapping),
                "symbols": int(mapping.symbol.nunique()), "dated": "date" in mapping}
    finally:
        if temporary.exists():
            temporary.unlink()


def save_factor_upload(filename: str, body: bytes) -> dict[str, Any]:
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise ValueError("因子文件仅支持.csv或.parquet")
    if not body or len(body) > 256 * 1024 * 1024:
        raise ValueError("因子文件必须大于0且不超过256MB")
    safe_name = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", Path(filename).stem).strip("_")
    if not safe_name:
        raise ValueError("因子文件名无效")
    FACTOR_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temporary = FACTOR_UPLOAD_DIR / f".upload-{os.getpid()}-{threading.get_ident()}.parquet"
    target = FACTOR_UPLOAD_DIR / f"{safe_name}.parquet"
    try:
        raw = FACTOR_UPLOAD_DIR / f".raw-{os.getpid()}-{threading.get_ident()}{suffix}"
        raw.write_bytes(body)
        frame = pd.read_csv(raw) if suffix == ".csv" else pd.read_parquet(raw).reset_index()
        raw.unlink(missing_ok=True)
        if not {"date", "ticker"}.issubset(frame.columns):
            if {"factor_date", "symbol"}.issubset(frame.columns):
                frame = frame.rename(columns={"factor_date": "date", "symbol": "ticker"})
            else:
                raise ValueError("因子文件必须包含 date、ticker（或 factor_date、symbol）")
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["ticker"] = frame["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
        if frame["date"].isna().any() or frame["ticker"].eq("").any():
            raise ValueError("因子文件包含无法解析的日期或股票代码")
        numeric = [column for column in frame.columns if column not in {"date", "ticker", *FEATURE_EXCLUDED_COLUMNS} and
                   pd.api.types.is_numeric_dtype(frame[column])]
        if not numeric:
            raise ValueError("因子文件至少需要一个数值型因子列")
        suspicious = [column for column in numeric if any(token in column.lower()
                     for token in FEATURE_FORBIDDEN_PATTERNS)]
        if suspicious:
            raise ValueError(f"检测到疑似未来信息特征：{suspicious}")
        if frame.duplicated(["date", "ticker"]).any():
            raise ValueError("因子文件存在重复的日期和股票组合")
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, target)
        return {"ok": True, "factor_file": target.name, "filename": target.name,
                "rows": len(frame), "features": numeric,
                "start": str(frame.date.min().date()), "end": str(frame.date.max().date())}
    finally:
        temporary.unlink(missing_ok=True)
        if 'raw' in locals():
            raw.unlink(missing_ok=True)


def validate_period_config(raw: dict[str, Any]) -> dict[str, str]:
    values = {"train_start": str(raw.get("train_start", "2020-01-01")),
              "train_end": str(raw.get("train_end", "2023-12-31")),
              "test_start": str(raw.get("test_start", raw.get("backtest_start", "2024-01-01"))),
              "test_end": str(raw.get("test_end", raw.get("backtest_end", "2024-12-31")))}
    dates = [pd.Timestamp(values[key]) for key in ("train_start", "train_end", "test_start", "test_end")]
    if not dates[0] <= dates[1] < dates[2] <= dates[3]:
        raise ValueError("训练区间必须早于测试区间，且日期顺序必须有效")
    return {key: str(pd.Timestamp(value).date()) for key, value in values.items()}


def validate_style_config(raw: Any) -> dict[str, int | float | str]:
    """验证网页传入的风格参数，拒绝静默使用越界值。"""
    source = raw if isinstance(raw, dict) else {}
    config = {
        "pca_variance": float(source.get("pca_variance", STYLE_CONFIG_DEFAULTS["pca_variance"])),
        "pca_min_components": int(source.get("pca_min_components", STYLE_CONFIG_DEFAULTS["pca_min_components"])),
        "xgb_max_depth": int(source.get("xgb_max_depth", STYLE_CONFIG_DEFAULTS["xgb_max_depth"])),
        "pca_alignment_threshold": float(source.get("pca_alignment_threshold", STYLE_CONFIG_DEFAULTS["pca_alignment_threshold"])),
        "pca_explained_threshold": float(source.get("pca_explained_threshold", STYLE_CONFIG_DEFAULTS["pca_explained_threshold"])),
        "confidence_rankic_window": int(source.get("confidence_rankic_window", STYLE_CONFIG_DEFAULTS["confidence_rankic_window"])),
        "confidence_rankic_threshold": float(source.get("confidence_rankic_threshold", STYLE_CONFIG_DEFAULTS["confidence_rankic_threshold"])),
        "holding_count": int(source.get("holding_count", STYLE_CONFIG_DEFAULTS["holding_count"])),
        "rotation_quantile": float(source.get("rotation_quantile", STYLE_CONFIG_DEFAULTS["rotation_quantile"])),
        "rebalance_frequency": str(source.get("rebalance_frequency", STYLE_CONFIG_DEFAULTS["rebalance_frequency"])),
        "pca_fit_mode": str(source.get("pca_fit_mode", STYLE_CONFIG_DEFAULTS["pca_fit_mode"])),
        "weight_mode": str(source.get("weight_mode", STYLE_CONFIG_DEFAULTS["weight_mode"])),
    }
    if not .50 <= config["pca_variance"] <= .999:
        raise ValueError("PCA累计解释率必须在0.50到0.999之间")
    if not 1 <= config["pca_min_components"] <= 100:
        raise ValueError("PCA最少成分数必须在1到100之间")
    if not 1 <= config["xgb_max_depth"] <= 10:
        raise ValueError("PCA-XGBoost最大深度必须在1到10之间")
    if not 0 <= config["pca_alignment_threshold"] <= 1:
        raise ValueError("PCA载荷平均相似度门槛必须在0到1之间")
    if not 0 <= config["pca_explained_threshold"] <= 1:
        raise ValueError("PCA交易解释率门槛必须在0到1之间")
    if not 10 <= config["confidence_rankic_window"] <= 1000:
        raise ValueError("已实现RankIC窗口必须在10到1000之间")
    if not -1 <= config["confidence_rankic_threshold"] <= 1:
        raise ValueError("RankIC门槛必须在-1到1之间")
    if config["holding_count"] < 1:
        raise ValueError("目标持仓数必须大于0")
    if not 0 < config["rotation_quantile"] <= 1:
        raise ValueError("调仓分位比例必须在0到1之间")
    if config["rebalance_frequency"] not in {"daily", "alternate", "weekly", "monthly", "quarterly"}:
        raise ValueError("调仓频率必须是日频、隔日、周频或月频")
    if config["pca_fit_mode"] not in {"rolling", "fixed"}:
        raise ValueError("PCA拟合方式必须是滚动训练或固定训练期")
    if config["weight_mode"] not in {"score", "equal"}:
        raise ValueError("组合权重方式必须是得分加权或等权")
    return config

    # Legacy style-timing validation retained below for old-run source compatibility.
    if not .50 <= config["pca_variance"] <= .999:
        raise ValueError("PCA累计解释率必须在0.50到0.999之间")
    if not 1 <= config["pca_min_components"] <= 100:
        raise ValueError("PCA最少成分数必须在1到100之间")
    if not 1 <= config["buy_confirmations"] <= 100:
        raise ValueError("买入确认风格数必须在1到100之间")
    if not .50 < config["vote_quantile"] < 1:
        raise ValueError("看多/看空分位必须在0.50到1之间")
    if not 0 <= config["residual_weight"] <= 10:
        raise ValueError("残差权重必须在0到10之间")
    if config["residual_alpha"] <= 0:
        raise ValueError("残差Ridge alpha必须大于0")
    if not 10 <= config["icir_window"] <= 1000:
        raise ValueError("滚动ICIR窗口必须在10到1000之间")
    if not 1 <= config["xgb_max_depth"] <= 10:
        raise ValueError("风格XGBoost最大深度必须在1到10之间")
    return config


def json_safe(value: Any) -> Any:
    """将NumPy/Pandas标量和NaN/Infinity转换为浏览器可解析的标准JSON值。"""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    return value


class TrainingJob:
    def __init__(self, recover: bool = False):
        self.lock = threading.RLock()
        self.state = "idle"
        self.current_model: str | None = None
        self.current_run_id: str | None = None
        self.models: list[str] = []
        self.completed_models: list[str] = []
        self.run_dirs: dict[str, str] = {}
        self.run_models: dict[str, str] = {}
        self.logs: list[str] = []
        self.process: subprocess.Popen | None = None
        self.process_pid: int | None = None
        self.stop_event = threading.Event()
        self.error = ""
        if recover:
            self._load_runs()
            self._recover_running_process()

    @property
    def registry_path(self) -> Path:
        return OUTPUT_ROOT / "run_registry.json"

    def _load_runs(self) -> None:
        payload = self._json(self.registry_path, [])
        items = payload if isinstance(payload, list) else []
        registered = {str(item.get("run_id", "")): item for item in items}
        # Completed run directories are the source of truth. Rebuild a missing
        # or stale registry after a service interruption instead of hiding
        # historical results from the frontend.
        pattern = re.compile(
            r"run_[0-9]{8}_[0-9]{6}(?:_[0-9]{6})?_[A-Za-z0-9_]+$")
        for path in OUTPUT_ROOT.glob("run_*") if OUTPUT_ROOT.exists() else []:
            if path.is_dir() and pattern.fullmatch(path.name):
                registered.setdefault(path.name, {"run_id": path.name, "model": ""})
        for item in registered.values():
            run_id, model = str(item.get("run_id", "")), str(item.get("model", ""))
            path = (OUTPUT_ROOT / run_id).resolve()
            if not model and path.is_dir():
                manifest = self._json(path / "config/run_manifest.json", {})
                models = manifest.get("models", []) if isinstance(manifest, dict) else []
                model = str(models[0]) if len(models) == 1 else ""
            if run_id and model and path.is_dir() and OUTPUT_ROOT.resolve() in path.parents:
                self.run_dirs[run_id], self.run_models[run_id] = str(path), model
        if self.run_dirs:
            self._save_runs()
    def _save_runs(self) -> None:
        payload = [{"run_id": run_id, "model": self.run_models.get(run_id, ""),
                    "path": directory} for run_id, directory in self.run_dirs.items()]
        self.registry_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _recover_running_process(self) -> None:
        """Web服务重启后，在Windows上重新挂接仍在运行的main_rolling进程。"""
        if os.name != "nt": return
        command = ("Get-CimInstance Win32_Process | Where-Object {$_.Name -eq 'python.exe' -and "
                   "$_.CommandLine -match 'main_rolling.py'} | Select-Object ProcessId,ParentProcessId,CommandLine | ConvertTo-Json")
        try:
            raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", command],
                                          text=True, encoding="utf-8", errors="replace", timeout=8).strip()
            if not raw: return
            processes = json.loads(raw)
            if isinstance(processes, dict): processes = [processes]
            # 虚拟环境启动器及真实解释器会出现两条记录，选择父级启动器以便终止整棵进程树。
            parent_ids = {int(item.get("ParentProcessId", 0)) for item in processes}
            candidates = [item for item in processes if int(item.get("ProcessId", 0)) in parent_ids] or processes
            item = candidates[0]
            line = item.get("CommandLine", "")
            run_match = re.search(r"--resume-run\s+(\S+)", line)
            model_match = re.search(r"--models\s+(\S+)", line)
            if not run_match or not model_match: return
            run_id, model = run_match.group(1), model_match.group(1)
            self.state, self.current_model, self.current_run_id = "running", model, run_id
            self.models, self.completed_models = [model], []
            self.run_dirs[run_id] = str(OUTPUT_ROOT / run_id)
            self.run_models[run_id] = model
            self._save_runs()
            self.process_pid = int(item["ProcessId"])
            self.logs = [f"Web服务已重新挂接正在运行的任务：{model}（PID {self.process_pid}）"]
        except Exception:
            return

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            payload = {"state": self.state, "current_model": self.current_model,
                       "current_run_id": self.current_run_id, "models": self.models,
                       "completed_models": self.completed_models, "run_dirs": self.run_dirs,
                       "logs": self.logs[-300:], "error": self.error}
            run_dirs = dict(self.run_dirs)
        model_results = {}
        for run_id, directory in run_dirs.items():
            run_dir = Path(directory)
            if run_dir.exists():
                model_results[run_id] = {"run_dir": directory,
                                         "model": self.run_models.get(run_id, run_id.rsplit("_", 1)[-1]),
                                         **self._artifacts(run_dir)}
        payload["model_results"] = model_results
        current = model_results.get(self.current_run_id or "", {})
        payload.update({key: current.get(key, default) for key, default in
                        {"live": {}, "training": [], "daily_ic": [], "component_ic": [], "trials": [], "tuning": [],
                         "backtest": {}, "backtests": {}, "backtest_metrics": [], "report": ""}.items()})
        return payload

    @staticmethod
    def _json(path: Path, default):
        try: return json.loads(path.read_text(encoding="utf-8"))
        except Exception: return default

    @staticmethod
    def _csv(path: Path) -> list[dict]:
        try:
            frame = pd.read_csv(path)
            return json.loads(frame.replace({np.nan: None}).to_json(orient="records", date_format="iso"))
        except Exception: return []

    def _artifacts(self, run_dir: Path) -> dict[str, Any]:
        live_files = sorted((run_dir / "backtest/results").glob("*_live.json")) if (run_dir / "backtest/results").exists() else []
        backtests = {}
        for live_file in live_files:
            payload = self._json(live_file, {})
            pool = payload.get("pool", live_file.stem)
            layer_file = live_file.parent / f"{payload.get('model', live_file.stem)}_{pool}_layers.csv"
            payload["layers"] = self._csv(layer_file)
            backtests[pool] = payload
        backtest_live = next(reversed(backtests.values()), {}) if backtests else {}
        trial_files = sorted((run_dir / "optuna").glob("*_trials_live.csv")) if (run_dir / "optuna").exists() else []
        tuning_files = sorted((run_dir / "metrics").glob("*_tuning_progress.csv")) if (run_dir / "metrics").exists() else []
        return {"live": self._json(run_dir / "logs/live_status.json", {}),
                "training": self._csv(run_dir / "metrics/training_records.csv"),
                "daily_ic": self._csv(run_dir / "metrics/daily_rankic.csv"),
                "component_ic": self._csv(run_dir / "metrics/score_component_rankic.csv"),
                "pca_style_ic": self._csv(run_dir / "metrics/pca_style_rankic.csv"),
                "trials": self._csv(trial_files[-1]) if trial_files else [],
                "tuning": self._csv(tuning_files[-1]) if tuning_files else [],
                "backtest": backtest_live,
                "backtests": backtests,
                "backtest_metrics": self._csv(run_dir / "backtest/results/summary.csv"),
                "report": str(run_dir / "report/index.html") if (run_dir / "report/index.html").exists() else ""}

    def start(self, config: dict[str, Any]) -> None:
        with self.lock:
            if self.state in {"running", "stopping"}:
                raise RuntimeError("已有任务正在运行")
            models = config.get("models") or []
            allowed = {"linear_regression", "ridge", "elasticnet", "xgboost", "style_rotation"}
            if not models or not set(models).issubset(allowed):
                raise ValueError("请选择合法模型")
            pools = config.get("pools") or []
            if not pools or not set(pools).issubset(set(SUPPORTED_POOLS)):
                raise ValueError("请至少选择一个合法票池")
            window_mode = str(config.get("window_mode", "expanding"))
            if window_mode not in {"expanding", "fixed"}:
                raise ValueError("普通模型训练窗口必须是扩展窗口或滚动固定窗口")
            # The public model form only exposes a backtest range.  Keep the
            # worker's historical train/test contract internally, deriving the
            # validation cutoff from the first backtest year.  Legacy callers
            # that still send train_* continue to work unchanged.
            backtest_start = str(config.get("backtest_start") or
                                 config.get("test_start", "2024-01-01"))
            backtest_end = str(config.get("backtest_end") or
                               config.get("test_end", "2024-12-31"))
            config["test_start"], config["test_end"] = backtest_start, backtest_end
            first_test = pd.Timestamp(backtest_start)
            auto_train_start = "train_start" not in config or not str(config.get("train_start", "")).strip()
            if window_mode == "fixed" or "train_end" not in config or "backtest_start" in config:
                config["train_end"] = f"{first_test.year - 1}-12-31"
            train_years = int(config.get("train_years", 4))
            if train_years < 1 or train_years > 20:
                raise ValueError("滚动训练年数必须在1到20之间")
            config.update({"window_mode": window_mode, "train_years": train_years})
            config["retune_each_year"] = bool(config.get("retune_each_year", False))
            factor_file = str(config.get("factor_file", "")).strip()
            rebalance_frequency = str(config.get("rebalance_frequency", "daily"))
            if rebalance_frequency not in {"daily", "alternate", "weekly", "monthly", "quarterly"}:
                raise ValueError("调仓频率必须是日频、隔日、周频或月频")
            config["rebalance_frequency"] = rebalance_frequency
            selection_quantile = float(config.get("selection_quantile", .10))
            if not 0 < selection_quantile <= 1:
                raise ValueError("普通模型选股分位必须在0到1之间")
            config["selection_quantile"] = selection_quantile
            legacy_has_optimizer = any(str(config.get(key, "none")) != "none" for key in
                                       ("optimizer", "turnover_optimizer", "drawdown_optimizer"))
            stage = str(config.get("backtest_stage", "optimized" if legacy_has_optimizer else "pure"))
            if stage not in {"pure", "cost", "optimized"}:
                raise ValueError("回测阶段必须是纯模型、交易成本或优化器")
            include_costs = stage in {"cost", "optimized"}
            commission_rate = float(config.get("commission_rate", .0003))
            stamp_duty_rate = float(config.get("stamp_duty_rate", .0005))
            if not 0 <= commission_rate <= .02 or not 0 <= stamp_duty_rate <= .02:
                raise ValueError("佣金和印花税率必须在0到2%之间")
            config.update({"backtest_stage": stage, "include_costs": include_costs,
                           "commission_rate": commission_rate,
                           "stamp_duty_rate": stamp_duty_rate})
            explicit_factor_source = "factor_source" in config
            # Monthly Calc is the default research library. Daily Alpha remains
            # selectable explicitly from the model page.
            factor_source = str(config.get("factor_source", "monthly_calc")).strip()
            # Backward compatibility for requests from an older frontend: if the
            # requested period predates daily Alpha coverage, prefer the monthly
            # built-in file instead of silently running the wrong universe.
            if not explicit_factor_source and factor_source == "daily_alpha":
                requested_start = pd.Timestamp(config["train_start"])
                daily_path = FACTOR_SOURCES["daily_alpha"]
                if requested_start < pd.Timestamp("2020-01-02") and FACTOR_SOURCES["monthly_calc"].is_file():
                    factor_source = "monthly_calc"
            factor_path = FACTOR_SOURCES.get(factor_source)
            if factor_file:
                candidate = (FACTOR_UPLOAD_DIR / Path(factor_file).name).resolve()
                if FACTOR_UPLOAD_DIR.resolve() not in candidate.parents or not candidate.is_file():
                    raise ValueError("所选因子文件不存在，请重新上传")
                factor_path = candidate
            elif factor_path is None or not factor_path.is_file():
                raise ValueError("所选内置因子文件不存在")
            config["factor_source"] = factor_source
            config["factor_path"] = str(factor_path)
            available_factors = factor_columns(factor_path)
            if "selected_factors" in config and not config.get("selected_factors"):
                raise ValueError("请至少选择一个模型因子")
            selected_factors = list(dict.fromkeys(
                str(name) for name in (config.get("selected_factors") or available_factors)))
            unknown_factors = [name for name in selected_factors if name not in available_factors]
            if unknown_factors:
                raise ValueError(f"所选因子不在当前因子文件中：{unknown_factors}")
            config["selected_factors"] = selected_factors
            config["factor_frequency"] = ("monthly" if factor_source == "monthly_calc"
                                           else "daily" if factor_source == "daily_alpha"
                                           else "custom")
            profile = LocalParquetSource(factor_path).inspect()
            available_start = pd.Timestamp(profile.start) if profile.start else None
            available_end = pd.Timestamp(profile.end) if profile.end else None
            if auto_train_start:
                if window_mode == "fixed":
                    derived = first_test - pd.DateOffset(years=train_years) + pd.Timedelta(days=1)
                    if available_start is not None:
                        derived = max(derived, available_start)
                else:
                    # Expanding mode deliberately starts at the first date in
                    # the selected factor library: every prediction period can
                    # use all available history before its cutoff.
                    derived = available_start or (first_test - pd.DateOffset(years=20))
                config["train_start"] = str(pd.Timestamp(derived).date())
            config.update(validate_period_config(config))
            requested_start, requested_end = pd.Timestamp(config["train_start"]), pd.Timestamp(config["test_end"])
            if available_start is not None and requested_start < available_start:
                raise ValueError(
                    f"训练开始日期早于因子文件覆盖范围：请求 {requested_start.date()}，"
                    f"因子最早 {available_start.date()}；请调整日期或选择更早覆盖的因子文件")
            if available_end is not None and requested_end > available_end:
                raise ValueError(
                    f"测试结束日期晚于因子文件覆盖范围：请求 {requested_end.date()}，"
                    f"因子最晚 {available_end.date()}；请调整日期或选择覆盖更晚的因子文件")
            config["backtest_start"], config["backtest_end"] = config["test_start"], config["test_end"]
            config["initial_train_end"] = config["train_end"]
            self.logs.append(f"因子文件：{factor_source} -> {factor_path}")
            self.logs.append(f"因子频率：{config['factor_frequency']}")
            self.logs.append(f"模型因子：{len(selected_factors)} 个 -> {', '.join(selected_factors)}")
            trials = int(config.get("optuna_trials", 30))
            if trials < 1 or trials > 10_000:
                raise ValueError("Optuna Trial必须在1到10000之间")
            if "style_rotation" in models:
                config["style_config"] = validate_style_config(config.get("style_config"))
            requested_optimizer = str(config.get("optimizer", "none"))
            if requested_optimizer not in OPTIMIZER_NAMES:
                raise ValueError("请选择合法优化器")
            optimizer = requested_optimizer if stage == "optimized" else "none"
            if optimizer != "none" and not config.get("skip_backtest"):
                create_optimizer(optimizer)
            config["optimizer"] = optimizer
            requested_turnover = str(config.get("turnover_optimizer", "none"))
            requested_drawdown = str(config.get("drawdown_optimizer", "none"))
            if requested_turnover not in TURNOVER_OPTIMIZER_NAMES:
                raise ValueError("请选择合法换手率优化器")
            if requested_drawdown not in DRAWDOWN_OPTIMIZER_NAMES:
                raise ValueError("请选择合法最大回撤优化器")
            turnover_optimizer = requested_turnover if stage == "optimized" else "none"
            drawdown_optimizer = requested_drawdown if stage == "optimized" else "none"
            if stage == "optimized" and all(name == "none" for name in
                                             (optimizer, turnover_optimizer, drawdown_optimizer)):
                raise ValueError("优化器阶段请至少选择换手率、最大回撤或行业中性化中的一项")
            sell_confirmations = int(config.get("sell_confirmations", TURNOVER_SELL_CONFIRMATIONS))
            max_turnover_ratio = float(config.get("max_turnover_ratio", TURNOVER_MAX_RATIO))
            max_drawdown_limit = float(config.get("max_drawdown_limit", DRAWDOWN_LIMIT))
            create_turnover_optimizer(turnover_optimizer, sell_confirmations, max_turnover_ratio)
            create_drawdown_optimizer(drawdown_optimizer, max_drawdown_limit)
            if turnover_optimizer == "lazy_turnover" and set(models) != {"style_rotation"}:
                raise ValueError("PCA置信度惰性持仓只能在仅选择PCA-XGBoost时启用")
            config.update({"turnover_optimizer": turnover_optimizer,
                           "drawdown_optimizer": drawdown_optimizer,
                           "sell_confirmations": sell_confirmations,
                           "max_turnover_ratio": max_turnover_ratio,
                           "max_drawdown_limit": max_drawdown_limit})
            self.state, self.models, self.completed_models = "running", list(models), []
            self.current_model, self.current_run_id, self.logs, self.error = None, None, [], ""
            self.stop_event.clear()
        threading.Thread(target=self._run, args=(config,), daemon=True, name="model-training-job").start()

    def _run(self, config: dict[str, Any]) -> None:
        try:
            for model in self.models:
                if self.stop_event.is_set(): break
                run_id = datetime.now().strftime(f"run_%Y%m%d_%H%M%S_%f_{model}")
                run_dir = OUTPUT_ROOT / run_id
                with self.lock:
                    self.current_model, self.current_run_id = model, run_id
                    self.run_dirs[run_id] = str(run_dir)
                    self.run_models[run_id] = model
                    self._save_runs()
                    self.logs.append(f"开始顺序任务：{model}")
                entry = "main_style_rotation.py" if model == "style_rotation" else "main_rolling.py"
                command = [sys.executable, str(MODEL_ROOT / entry), "--resume-run", run_id,
                           "--test-year", str(int(config.get("test_year", 2024))),
                           "--train-start", str(config["train_start"]),
                           "--train-end", str(config["train_end"]),
                           "--test-start", str(config["test_start"]),
                           "--test-end", str(config["test_end"]),
                           "--factor-path", str(config["factor_path"]),
                           "--optuna-trials", str(int(config.get("optuna_trials", 30))), "--n-jobs", str(MODEL_N_JOBS),
                           "--pools", *config.get("pools", ["A500", "ZZ1000"]),
                           "--optimizer", str(config.get("optimizer", "none")),
                           "--turnover-optimizer", str(config.get("turnover_optimizer", "none")),
                           "--drawdown-optimizer", str(config.get("drawdown_optimizer", "none")),
                           "--sell-confirmations", str(config.get("sell_confirmations", TURNOVER_SELL_CONFIRMATIONS)),
                           "--max-turnover-ratio", str(config.get("max_turnover_ratio", TURNOVER_MAX_RATIO)),
                           "--max-drawdown-limit", str(config.get("max_drawdown_limit", DRAWDOWN_LIMIT)),
                           "--commission-rate", str(config.get("commission_rate", .0003)),
                           "--stamp-duty-rate", str(config.get("stamp_duty_rate", .0005))]
                command.extend(["--features", *config["selected_factors"]])
                if config.get("include_costs"):
                    command.append("--include-costs")
                if model != "style_rotation":
                    command.extend(["--models", model, "--window-mode", str(config["window_mode"]),
                                    "--train-years", str(config["train_years"])])
                    if config.get("retune_each_year"):
                        command.append("--retune-each-year")
                    execution_style = config.get("style_config") or {}
                    command.extend([
                        "--holding-count", str(int(execution_style.get("holding_count", 200))),
                        "--selection-mode", "top_quantile",
                        "--selection-quantile", str(config.get("selection_quantile", .10)),
                        "--rebalance-frequency", str(config.get("rebalance_frequency", "daily")),
                        "--weight-mode", str(execution_style.get("weight_mode", "score")),
                    ])
                else:
                    # 全市场拟合滚动PCA，各股票池用PCA暴露直接训练股票收益XGBoost。
                    command.extend(["--style-models", "pca_stock_xgboost"])
                    command.extend(["--window-mode", str(config["window_mode"]),
                                    "--train-years", str(config["train_years"])])
                    style = validate_style_config(config.get("style_config"))
                    command.extend([
                        "--pca-variance", str(style["pca_variance"]),
                        "--pca-min-components", str(style["pca_min_components"]),
                        "--xgb-max-depth", str(style["xgb_max_depth"]),
                        "--pca-alignment-threshold", str(style["pca_alignment_threshold"]),
                        "--pca-explained-threshold", str(style["pca_explained_threshold"]),
                        "--confidence-rankic-window", str(style["confidence_rankic_window"]),
                        "--confidence-rankic-threshold", str(style["confidence_rankic_threshold"]),
                        "--holding-count", str(style["holding_count"]),
                        "--rotation-quantile", str(style["rotation_quantile"]),
                        "--rebalance-frequency", str(style["rebalance_frequency"]),
                        "--pca-fit-mode", str(style["pca_fit_mode"]),
                        "--weight-mode", str(style.get("weight_mode", "score")),
                    ])
                if config.get("skip_backtest"):
                    command.append("--skip-backtest")
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
                child_env = os.environ.copy()
                child_env["PYTHONIOENCODING"] = "utf-8"
                child_env["PYTHONUTF8"] = "1"
                child_env["OMP_NUM_THREADS"] = str(MODEL_N_JOBS)
                child_env["MKL_NUM_THREADS"] = str(MODEL_N_JOBS)
                child_env["OPENBLAS_NUM_THREADS"] = str(MODEL_N_JOBS)
                child_env["NUMEXPR_NUM_THREADS"] = str(MODEL_N_JOBS)
                # Python 3.13/原生扩展在Windows上偶发0xC0000409 fast-fail；仅对此码重试一次。
                code = 0
                for attempt in range(2):
                    launched_process = subprocess.Popen(command, cwd=MODEL_ROOT, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                        bufsize=1, creationflags=creationflags, env=child_env)
                    self.process = launched_process
                    self.process_pid = launched_process.pid
                    # Stop can arrive between Popen() and PID registration.
                    # Honor it immediately so a worker never survives a global
                    # stop request because of that small race window.
                    if self.stop_event.is_set():
                        self._terminate_process_tree(launched_process, launched_process.pid)
                    assert launched_process.stdout is not None
                    for line in launched_process.stdout:
                        with self.lock: self.logs.append(line.rstrip())
                        if self.stop_event.is_set(): break
                    code = launched_process.wait()
                    with self.lock:
                        if self.process is launched_process:
                            self.process = None
                            self.process_pid = None
                    if code != 3221226505 or self.stop_event.is_set() or attempt == 1:
                        break
                    with self.lock:
                        self.logs.append(f"{model} 遇到Windows原生退出码0xC0000409，自动重试一次")
                if self.stop_event.is_set(): break
                if code != 0:
                    raise RuntimeError(f"{model} 退出码：{code}")
                with self.lock:
                    self.completed_models.append(run_id)
                    self.logs.append(f"完成：{model}")
            with self.lock:
                self.state = "stopped" if self.stop_event.is_set() else "complete"
        except Exception as exc:
            with self.lock:
                self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"
                self.logs.append(self.error)

    def delete_run(self, run_id: str) -> None:
        with self.lock:
            if not re.fullmatch(r"run_[0-9]{8}_[0-9]{6}(?:_[0-9]{6})?_[A-Za-z0-9_]+", run_id):
                raise ValueError("运行ID无效")
            if self.current_run_id == run_id and self.state in {"running", "stopping"}:
                raise RuntimeError("当前运行中的结果不能删除")
            directory = self.run_dirs.get(run_id)
            if directory is None:
                raise FileNotFoundError("运行结果不存在")
            target = Path(directory).resolve()
            if target.parent != OUTPUT_ROOT.resolve() or target.name != run_id:
                raise ValueError("运行目录不安全")
            if target.exists():
                shutil.rmtree(target)
            self.run_dirs.pop(run_id, None)
            self.run_models.pop(run_id, None)
            self.completed_models = [item for item in self.completed_models if item != run_id]
            self._save_runs()

    def stop(self) -> None:
        with self.lock:
            if self.state not in {"running", "stopping"}: return
            self.state = "stopping"
            self.stop_event.set()
            process = self.process
            process_pid = self.process_pid
        self._terminate_process_tree(process, process_pid)
        # 恢复挂接的任务没有本服务创建的调度线程，必须在这里完成状态收尾。
        with self.lock:
            self.process = None
            self.process_pid = None
            self.state = "stopped"
            self.logs.append("任务已终止，模型进程树已结束")

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen | None, process_pid: int | None) -> None:
        """Terminate the worker and all native/parallel child processes."""
        pid = process.pid if process is not None else process_pid
        if not pid:
            return
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        elif process is not None and process.poll() is None:
            process.terminate()


JOB = TrainingJob(recover=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): return

    def _send(self, body: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def _json_response(self, payload: Any, status: int = 200):
        body = json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send(body, "application/json; charset=utf-8", status)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/status": return self._json_response(JOB.snapshot())
        if path == "/api/optimizer-info": return self._json_response(optimizer_info())
        if path == "/api/framework": return self._json_response(RESEARCH.metadata())
        if path == "/plotly.min.js":
            import plotly
            file = Path(plotly.__file__).parent / "package_data/plotly.min.js"
            return self._send(file.read_bytes(), "application/javascript; charset=utf-8")
        file = WEB_ROOT / ("index.html" if path == "/" else path.lstrip("/"))
        if file.exists() and file.is_file() and WEB_ROOT.resolve() in file.resolve().parents:
            content_type = "text/html; charset=utf-8" if file.suffix == ".html" else "text/plain; charset=utf-8"
            return self._send(file.read_bytes(), content_type)
        self._json_response({"error": "not found"}, 404)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            parsed = urlparse(self.path)
            if parsed.path == "/api/industry-upload":
                if length > 100 * 1024 * 1024:
                    raise ValueError("行业分类文件不能超过100MB")
                filename = parse_qs(parsed.query).get("filename", [""])[0]
                return self._json_response(save_industry_upload(filename, self.rfile.read(length)))
            if parsed.path == "/api/factor-upload":
                filename = parse_qs(parsed.query).get("filename", [""])[0]
                return self._json_response(save_factor_upload(filename, self.rfile.read(length)), 201)
            payload = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/start":
                JOB.start(payload); return self._json_response({"ok": True})
            if parsed.path == "/api/factor-analysis":
                return self._json_response(RESEARCH.analyze_factor(payload))
            if parsed.path == "/api/stop":
                JOB.stop(); return self._json_response({"ok": True})
            if parsed.path == "/api/delete-run":
                JOB.delete_run(str(payload.get("run_id", "")))
                return self._json_response({"ok": True})
            self._json_response({"error": "not found"}, 404)
        except Exception as exc:
            self._json_response({"error": str(exc)}, 400)


def main() -> None:
    parser = argparse.ArgumentParser(description="模型训练实时控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"模型控制台：http://{args.host}:{args.port}")
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        JOB.stop(); server.server_close()


if __name__ == "__main__": main()
