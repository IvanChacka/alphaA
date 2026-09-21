"""滚动机器学习全部配置。命令行只覆盖这里的值。"""
from imports import *

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "BackTestData"
ADJ_CLOSE_MARKET_PATH = DATA_DIR / "adj_close.pq" / "adj_close.pq" / "adj_close.parquet"
BACKTEST_DIR = PROJECT_ROOT / "code" / "BackTest"
OUTPUT_ROOT = MODEL_ROOT / "output"

FACTOR_PATH = DATA_DIR / "factordata.parquet"
MONTHLY_FACTOR_PATH = DATA_DIR / "data_in_sample_monthly.parquet" / "data_in_sample_monthly.parquet"
FACTOR_SOURCES = {
    "daily_alpha": FACTOR_PATH,
    "monthly_calc": MONTHLY_FACTOR_PATH,
}
TWAP_PATH = DATA_DIR / "twap.parquet"
ADJ_FACTOR_PATH = DATA_DIR / "accumAdjFactor.parquet"
CALENDAR_PATH = DATA_DIR / "calendar.csv"

ML_TRAIN_START = "2020-01-01"
ML_INITIAL_TRAIN_END = "2023-12-31"
ML_TEST_START = "2024-01-01"
ML_TEST_END = "2024-12-31"
ROLLING_FREQUENCY = "quarter"
HYPERPARAMETER_TUNING_FREQUENCY = "year"
ROLLING_WINDOW_MODE = "expanding"
ROLLING_TRAIN_YEARS = 4
LINEAR_MODELS = ["linear_regression", "ridge", "elasticnet"]
NONLINEAR_MODELS = ["xgboost"]
OPTUNA_N_TRIALS = 30
OPTUNA_N_STARTUP_TRIALS = 20
OPTUNA_SEED = 42
MODEL_N_JOBS = max(1, os.cpu_count() or 1)
# Trial 保持顺序执行，单个 XGBoost 模型使用全部逻辑核，避免 Trial
# 并行造成嵌套过度并行。前端任务按模型顺序运行，因此不会同时启动多个 XGB。
OPTUNA_N_JOBS = 1
XGBOOST_N_JOBS = MODEL_N_JOBS
MODEL_RANDOM_SEED = 42
RANKIC_METHOD = "spearman"
ANNUAL_TRADING_DAYS = 252
SUPPORTED_POOLS = ["A500", "ZZ1000", "ALL_A500", "ALL_ZZ1000", "ALL_MARKET"]
ACTIVE_POOLS = ["A500", "ZZ1000"]

OPTIMIZER_NAMES = ["none", "industry_neutral"]
TURNOVER_OPTIMIZER_NAMES = ["none", "turnover_limit", "lazy_turnover"]
DRAWDOWN_OPTIMIZER_NAMES = ["none", "quadratic_drawdown", "max_drawdown"]
TURNOVER_SELL_CONFIRMATIONS = 2
TURNOVER_MAX_RATIO = 0.30
DRAWDOWN_LIMIT = 0.20
DRAWDOWN_QP_LOOKBACK = 60
DRAWDOWN_QP_MIN_OBSERVATIONS = 10
DRAWDOWN_QP_MIN_EXPOSURE = 0.30
DRAWDOWN_QP_RISK_AVERSION = 4.0
DRAWDOWN_QP_TURNOVER_PENALTY = 0.0025
DRAWDOWN_QP_RETURN_REWARD = 1.0
DRAWDOWN_QP_MAX_EXPOSURE_STEP = 0.10
INDUSTRY_DATA_CANDIDATES = [DATA_DIR / "industry.parquet", DATA_DIR / "industry.csv"]

REPORT_MAX_TABLE_ROWS = 3000
FEATURE_FORBIDDEN_PATTERNS = ("future", "forward", "lead", "next_", "t+1", "t+2", "未来")
FEATURE_EXCLUDED_COLUMNS = {"r_shift"}
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
ELASTICNET_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0]
ELASTICNET_L1_RATIO_GRID = [0.2, 0.8]
ELASTICNET_VALIDATION_FOLDS = 2
XGB_OBJECTIVE = "reg:pseudohubererror"
XGB_N_ESTIMATORS = 200
XGB_VALIDATION_FOLDS = 2
XGB_EARLY_STOPPING_ROUNDS = 100

# 滚动风格提取与风格择时。目标季度只使用此前两个完整季度拟合PCA。
STYLE_PCA_START = "2020-01-01"
STYLE_PCA_WINDOW_QUARTERS = 2
STYLE_VALIDATION_YEAR = 2022
STYLE_SELECTION_YEAR = 2023
STYLE_TEST_YEAR = 2024
STYLE_EXPLAINED_VARIANCE = 0.90
STYLE_MIN_COMPONENTS = 5
STYLE_QUANTILE = 0.20
STYLE_LABEL_HORIZON = 5
STYLE_TIME_DECAY_HALFLIFE = 252
STYLE_MODELS = ["equal_weight", "momentum_rule", "multinomial_logistic",
                "xgboost_classifier", "xgboost_regressors", "xgboost_dual_horizon",
                "pca_stock_xgboost"]
STYLE_ICIR_WINDOW = 60
STYLE_BUY_CONFIRMATIONS = 1
STYLE_VOTE_QUANTILE = 0.90
STYLE_RESIDUAL_WEIGHT = 1.0
STYLE_RESIDUAL_RIDGE_ALPHA = 0.05
STYLE_TIMING_TRAIN_START = "2022-01-01"
STYLE_XGB_MAX_ESTIMATORS = 200
STYLE_XGB_EARLY_STOPPING_ROUNDS = 20
STYLE_XGB_VALIDATION_EMBARGO_DAYS = 20
STYLE_XGB_MAX_DEPTH = 4
STYLE_PCA_ALIGNMENT_THRESHOLD = 0.80
STYLE_PCA_EXPLAINED_THRESHOLD = 0.85
STYLE_CONFIDENCE_RANKIC_WINDOW = 60
STYLE_CONFIDENCE_RANKIC_THRESHOLD = 0.0
STYLE_ENTRY_RANK = 200
STYLE_EXIT_RANK = 300
STYLE_HOLDING_COUNT = 200
STYLE_ROTATION_QUANTILE = 0.10
STYLE_REBALANCE_FREQUENCY = "daily"


@dataclass(frozen=True)
class MLConfig:
    train_start: str = ML_TRAIN_START
    initial_train_end: str = ML_INITIAL_TRAIN_END
    test_start: str = ML_TEST_START
    test_end: str = ML_TEST_END
    window_mode: str = ROLLING_WINDOW_MODE
    train_years: int = ROLLING_TRAIN_YEARS
    models: tuple[str, ...] = tuple(LINEAR_MODELS + NONLINEAR_MODELS)
    pools: tuple[str, ...] = tuple(ACTIVE_POOLS)
    optuna_trials: int = OPTUNA_N_TRIALS
    n_jobs: int = 1
    seed: int = MODEL_RANDOM_SEED
    output_root: Path = OUTPUT_ROOT
    factor_path: Path = FACTOR_PATH
    selected_features: tuple[str, ...] = ()
    rebalance_frequency: str = "daily"
    retune_each_year: bool = False

    def validate(self) -> None:
        if self.rebalance_frequency not in {"daily", "alternate", "weekly", "monthly", "quarterly"}:
            raise ValueError("调仓频率必须是日频、隔日、周频或月频")
        if self.train_years < 1 or self.train_years > 20:
            raise ValueError("训练窗口年数必须在1到20之间")
        if self.window_mode not in {"expanding", "fixed"}:
            raise ValueError("window_mode必须是 expanding 或 fixed")
        unknown = set(self.models) - set(LINEAR_MODELS + NONLINEAR_MODELS)
        if unknown:
            raise ValueError(f"未知模型：{sorted(unknown)}")
        dates = [pd.Timestamp(self.train_start), pd.Timestamp(self.initial_train_end),
                 pd.Timestamp(self.test_start), pd.Timestamp(self.test_end)]
        if not dates[0] <= dates[1] < dates[2] <= dates[3]:
            raise ValueError("训练区间必须早于测试区间，且日期顺序必须有效")
        if not Path(self.factor_path).exists():
            raise ValueError(f"因子文件不存在：{self.factor_path}")
