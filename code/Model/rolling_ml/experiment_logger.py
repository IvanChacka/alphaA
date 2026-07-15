from imports import *
from env import OUTPUT_ROOT


class ExperimentLogger:
    """统一管理一次运行的目录、进度日志和失败记录。"""

    def __init__(self, output_root: Path = OUTPUT_ROOT, run_id: str | None = None):
        self.run_id = run_id or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.root = Path(output_root) / self.run_id
        names = ["config", "audit", "optuna", "models", "predictions/quarterly",
                 "predictions/model_prediction_files", "metrics", "backtest/results",
                 "backtest/trades", "backtest/holdings", "backtest/adjustments", "logs", "report"]
        for name in names:
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.run_id)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(self.root / "logs/rolling_training.log", encoding="utf-8")
        stream_handler = logging.StreamHandler(sys.stdout)
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        self.failures: list[dict[str, Any]] = []
        self._status: dict[str, Any] = {"run_id": self.run_id, "stage": "starting", "progress": 0,
                                       "message": "运行初始化", "updated_at": datetime.now().isoformat()}
        self.json("logs/live_status.json", self._status)
        self.info("运行开始：%s", self.root)

    def info(self, message: str, *args: Any) -> None:
        self.logger.info(message, *args)

    def status(self, stage: str, progress: float, message: str, **extra: Any) -> None:
        self._status.update({"stage": stage, "progress": max(0.0, min(1.0, float(progress))),
                             "message": message, "updated_at": datetime.now().isoformat(), **extra})
        self.json("logs/live_status.json", self._status)

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)

    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb): self.close()

    @staticmethod
    def _json_value(value: Any):
        if isinstance(value, np.integer): return int(value)
        if isinstance(value, np.floating): return float(value)
        if isinstance(value, (Path, pd.Timestamp)): return str(value)
        if isinstance(value, tuple): return list(value)
        raise TypeError(type(value).__name__)

    def json(self, relative: str, payload: Any) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=self._json_value), encoding="utf-8")

    def csv(self, relative: str, frame: pd.DataFrame) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")

    def record_failure(self, model: str, period: str, exc: BaseException,
                       params: dict | None = None, train_range: str = "") -> None:
        item = {"model": model, "period": period, "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
                "parameters": json.dumps(params or {}, ensure_ascii=False), "train_range": train_range}
        self.failures.append(item)
        self.logger.exception("%s %s 失败", model, period)
        self.csv("logs/failed_tasks.csv", pd.DataFrame(self.failures))
