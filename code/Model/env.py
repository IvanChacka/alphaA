"""滚动机器学习全部配置。命令行只覆盖这里的值。"""
from imports import *

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parents[1]
DATA_DIR = PROJECT_ROOT / "BackTestData"
BACKTEST_DIR = PROJECT_ROOT / "code" / "BackTest"
OUTPUT_ROOT = MODEL_ROOT / "output"

FACTOR_PATH = DATA_DIR / "factordata.parquet"
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
MODEL_N_JOBS = 12
# Trial保持顺序执行，单个模型使用12线程，避免Trial并行造成嵌套过度并行。
OPTUNA_N_JOBS = 1
XGBOOST_N_JOBS = MODEL_N_JOBS
MODEL_RANDOM_SEED = 42
RANKIC_METHOD = "spearman"
ANNUAL_TRADING_DAYS = 252
SUPPORTED_POOLS = ["A500", "ZZ1000", "ALL_A500", "ALL_ZZ1000"]
ACTIVE_POOLS = ["A500", "ZZ1000"]
OPTIMIZER_NAMES = ["none", "industry_neutral"]
INDUSTRY_DATA_CANDIDATES = [DATA_DIR / "industry.parquet", DATA_DIR / "industry.csv"]
REPORT_MAX_TABLE_ROWS = 3000
FEATURE_FORBIDDEN_PATTERNS = ("future", "forward", "lead", "next_", "t+1", "t+2", "未来")
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
ELASTICNET_ALPHA_GRID = [0.001, 0.01, 0.1, 1.0]
ELASTICNET_L1_RATIO_GRID = [0.2, 0.8]
ELASTICNET_VALIDATION_FOLDS = 2
XGB_OBJECTIVE = "reg:pseudohubererror"
XGB_N_ESTIMATORS = 200
XGB_VALIDATION_FOLDS = 2
XGB_EARLY_STOPPING_ROUNDS = 100

# 固定风格提取与风格择时。PCA只允许在该区间拟合，之后永久冻结。
STYLE_PCA_START = "2020-01-01"
STYLE_PCA_END = "2021-12-31"
STYLE_VALIDATION_YEAR = 2022
STYLE_SELECTION_YEAR = 2023
STYLE_TEST_YEAR = 2024
STYLE_EXPLAINED_VARIANCE = 0.90
STYLE_MIN_COMPONENTS = 5
STYLE_QUANTILE = 0.20
STYLE_LABEL_HORIZON = 5
STYLE_TIME_DECAY_HALFLIFE = 252
STYLE_MODELS = ["equal_weight", "momentum_rule", "multinomial_logistic",
                "xgboost_classifier", "xgboost_regressors", "xgboost_dual_horizon"]
STYLE_ICIR_WINDOW = 60
STYLE_BUY_CONFIRMATIONS = 1
STYLE_SELL_CONFIRMATIONS = 2
STYLE_VOTE_QUANTILE = 0.90
STYLE_RESIDUAL_WEIGHT = 1.0
STYLE_RESIDUAL_RIDGE_ALPHA = 0.05
STYLE_TIMING_TRAIN_START = "2022-01-01"
STYLE_XGB_MAX_ESTIMATORS = 200
STYLE_XGB_EARLY_STOPPING_ROUNDS = 20
STYLE_XGB_VALIDATION_EMBARGO_DAYS = 20


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

    def validate(self) -> None:
        if self.window_mode not in {"expanding", "fixed"}:
            raise ValueError("window_mode必须是 expanding 或 fixed")
        unknown = set(self.models) - set(LINEAR_MODELS + NONLINEAR_MODELS)
        if unknown:
            raise ValueError(f"未知模型：{sorted(unknown)}")
