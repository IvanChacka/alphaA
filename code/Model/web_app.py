"""滚动模型实时控制台：顺序执行模型、展示进度并支持终止。"""
from imports import *
from urllib.parse import parse_qs
from env import (DATA_DIR, INDUSTRY_DATA_CANDIDATES, MODEL_N_JOBS, MODEL_ROOT,
                 OPTIMIZER_NAMES, OUTPUT_ROOT, SUPPORTED_POOLS)
from optimizers import create_optimizer
from optimizers.industry_neutral import IndustryNeutralOptimizer


WEB_ROOT = MODEL_ROOT / "web"


STYLE_CONFIG_DEFAULTS = {
    "pca_variance": 0.90,
    "pca_min_components": 5,
    "buy_confirmations": 1,
    "sell_confirmations": 2,
    "vote_quantile": 0.90,
    "residual_weight": 1.0,
    "residual_alpha": 0.05,
    "icir_window": 60,
}


def optimizer_info() -> dict[str, Any]:
    existing = next((Path(path) for path in INDUSTRY_DATA_CANDIDATES if Path(path).exists()), None)
    payload = {"optimizers": OPTIMIZER_NAMES, "industry_file": str(existing) if existing else "",
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


def validate_style_config(raw: Any) -> dict[str, int | float]:
    """验证网页传入的风格参数，拒绝静默使用越界值。"""
    source = raw if isinstance(raw, dict) else {}
    config = {
        "pca_variance": float(source.get("pca_variance", STYLE_CONFIG_DEFAULTS["pca_variance"])),
        "pca_min_components": int(source.get("pca_min_components", STYLE_CONFIG_DEFAULTS["pca_min_components"])),
        "buy_confirmations": int(source.get("buy_confirmations", STYLE_CONFIG_DEFAULTS["buy_confirmations"])),
        "sell_confirmations": int(source.get("sell_confirmations", STYLE_CONFIG_DEFAULTS["sell_confirmations"])),
        "vote_quantile": float(source.get("vote_quantile", STYLE_CONFIG_DEFAULTS["vote_quantile"])),
        "residual_weight": float(source.get("residual_weight", STYLE_CONFIG_DEFAULTS["residual_weight"])),
        "residual_alpha": float(source.get("residual_alpha", STYLE_CONFIG_DEFAULTS["residual_alpha"])),
        "icir_window": int(source.get("icir_window", STYLE_CONFIG_DEFAULTS["icir_window"])),
    }
    if not .50 <= config["pca_variance"] <= .999:
        raise ValueError("PCA累计解释率必须在0.50到0.999之间")
    if not 1 <= config["pca_min_components"] <= 100:
        raise ValueError("PCA最少成分数必须在1到100之间")
    if not 1 <= config["buy_confirmations"] <= 100 or not 1 <= config["sell_confirmations"] <= 100:
        raise ValueError("买入/卖出确认风格数必须在1到100之间")
    if not .50 < config["vote_quantile"] < 1:
        raise ValueError("看多/看空分位必须在0.50到1之间")
    if not 0 <= config["residual_weight"] <= 10:
        raise ValueError("残差权重必须在0到10之间")
    if config["residual_alpha"] <= 0:
        raise ValueError("残差Ridge alpha必须大于0")
    if not 10 <= config["icir_window"] <= 1000:
        raise ValueError("滚动ICIR窗口必须在10到1000之间")
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
        self.models: list[str] = []
        self.completed_models: list[str] = []
        self.run_dirs: dict[str, str] = {}
        self.logs: list[str] = []
        self.process: subprocess.Popen | None = None
        self.process_pid: int | None = None
        self.stop_event = threading.Event()
        self.error = ""
        if recover:
            self._recover_running_process()

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
            self.state, self.current_model = "running", model
            self.models, self.completed_models = [model], []
            self.run_dirs = {model: str(OUTPUT_ROOT / run_id)}
            self.process_pid = int(item["ProcessId"])
            self.logs = [f"Web服务已重新挂接正在运行的任务：{model}（PID {self.process_pid}）"]
        except Exception:
            return

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            payload = {"state": self.state, "current_model": self.current_model, "models": self.models,
                       "completed_models": self.completed_models, "run_dirs": self.run_dirs,
                       "logs": self.logs[-300:], "error": self.error}
            run_dirs = dict(self.run_dirs)
        model_results = {}
        for model, directory in run_dirs.items():
            run_dir = Path(directory)
            if run_dir.exists():
                model_results[model] = {"run_dir": directory, **self._artifacts(run_dir)}
        payload["model_results"] = model_results
        current = model_results.get(self.current_model or "", {})
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
            backtests[pool] = payload
        backtest_live = next(reversed(backtests.values()), {}) if backtests else {}
        trial_files = sorted((run_dir / "optuna").glob("*_trials_live.csv")) if (run_dir / "optuna").exists() else []
        tuning_files = sorted((run_dir / "metrics").glob("*_tuning_progress.csv")) if (run_dir / "metrics").exists() else []
        return {"live": self._json(run_dir / "logs/live_status.json", {}),
                "training": self._csv(run_dir / "metrics/training_records.csv"),
                "daily_ic": self._csv(run_dir / "metrics/daily_rankic.csv"),
                "component_ic": self._csv(run_dir / "metrics/score_component_rankic.csv"),
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
            trials = int(config.get("optuna_trials", 30))
            if trials < 1 or trials > 10_000:
                raise ValueError("Optuna Trial必须在1到10000之间")
            if "style_rotation" in models:
                config["style_config"] = validate_style_config(config.get("style_config"))
            optimizer = str(config.get("optimizer", "none"))
            if optimizer not in OPTIMIZER_NAMES:
                raise ValueError("请选择合法优化器")
            if optimizer != "none" and not config.get("skip_backtest"):
                create_optimizer(optimizer)
            config["optimizer"] = optimizer
            self.state, self.models, self.completed_models = "running", list(models), []
            self.current_model, self.run_dirs, self.logs, self.error = None, {}, [], ""
            self.stop_event.clear()
        threading.Thread(target=self._run, args=(config,), daemon=True, name="model-training-job").start()

    def _run(self, config: dict[str, Any]) -> None:
        try:
            for model in self.models:
                if self.stop_event.is_set(): break
                run_id = datetime.now().strftime(f"run_%Y%m%d_%H%M%S_{model}")
                run_dir = OUTPUT_ROOT / run_id
                with self.lock:
                    self.current_model = model
                    self.run_dirs[model] = str(run_dir)
                    self.logs.append(f"开始顺序任务：{model}")
                entry = "main_style_rotation.py" if model == "style_rotation" else "main_rolling.py"
                command = [sys.executable, str(MODEL_ROOT / entry), "--resume-run", run_id,
                           "--test-year", str(int(config.get("test_year", 2024))),
                           "--optuna-trials", str(int(config.get("optuna_trials", 30))), "--n-jobs", str(MODEL_N_JOBS),
                           "--pools", *config.get("pools", ["A500", "ZZ1000"]),
                           "--optimizer", str(config.get("optimizer", "none"))]
                if model != "style_rotation":
                    command.extend(["--models", model])
                else:
                    # 网页正式风格策略只运行PCA风格 + 动量特征 + XGBoost分类择时。
                    command.extend(["--style-models", "xgboost_dual_horizon"])
                    style = validate_style_config(config.get("style_config"))
                    command.extend([
                        "--pca-variance", str(style["pca_variance"]),
                        "--pca-min-components", str(style["pca_min_components"]),
                        "--buy-confirmations", str(style["buy_confirmations"]),
                        "--sell-confirmations", str(style["sell_confirmations"]),
                        "--vote-quantile", str(style["vote_quantile"]),
                        "--residual-weight", str(style["residual_weight"]),
                        "--residual-alpha", str(style["residual_alpha"]),
                        "--icir-window", str(style["icir_window"]),
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
                    self.process = subprocess.Popen(command, cwd=MODEL_ROOT, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                        bufsize=1, creationflags=creationflags, env=child_env)
                    self.process_pid = self.process.pid
                    assert self.process.stdout is not None
                    for line in self.process.stdout:
                        with self.lock: self.logs.append(line.rstrip())
                        if self.stop_event.is_set(): break
                    code = self.process.wait()
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
                    self.completed_models.append(model)
                    self.logs.append(f"完成：{model}")
            with self.lock:
                self.state = "stopped" if self.stop_event.is_set() else "complete"
        except Exception as exc:
            with self.lock:
                self.state, self.error = "failed", f"{type(exc).__name__}: {exc}"
                self.logs.append(self.error)

    def stop(self) -> None:
        with self.lock:
            if self.state not in {"running", "stopping"}: return
            self.state = "stopping"
            self.stop_event.set()
            process = self.process
            process_pid = self.process_pid
        if process and process.poll() is None:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                process.terminate()
        elif process_pid and os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process_pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        # 恢复挂接的任务没有本服务创建的调度线程，必须在这里完成状态收尾。
        with self.lock:
            self.process = None
            self.process_pid = None
            self.state = "stopped"
            self.logs.append("任务已终止，模型进程树已结束")


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
            payload = json.loads(self.rfile.read(length) or b"{}")
            if parsed.path == "/api/start":
                JOB.start(payload); return self._json_response({"ok": True})
            if parsed.path == "/api/stop":
                JOB.stop(); return self._json_response({"ok": True})
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
