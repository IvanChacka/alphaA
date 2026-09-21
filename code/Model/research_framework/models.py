"""Model formulas, parameters and framework metadata for the frontend."""
from __future__ import annotations

from typing import Any


MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "linear_regression": {
        "name": "Linear", "family": "线性模型", "formula": "y = beta0 + sum(beta_i * z_i) + epsilon",
        "method": "最小二乘直接拟合下一持有期收益。系数正负给出因子方向。",
        "parameters": [{"key": "fit_intercept", "label": "截距", "type": "boolean", "default": True}],
    },
    "ridge": {
        "name": "Ridge", "family": "线性模型", "formula": "argmin ||y-Xbeta||^2 + alpha||beta||^2",
        "method": "L2 正则压缩共线因子系数；alpha 由时序验证选择。",
        "parameters": [{"key": "alpha", "label": "L2 强度", "type": "number", "default": 1.0, "min": 0}],
    },
    "elasticnet": {
        "name": "ElasticNet", "family": "线性模型", "formula": "argmin L + alpha[rho||beta||1 + (1-rho)||beta||2^2]",
        "method": "结合 L1 筛选和 L2 稳定性，适合高维相关因子。",
        "parameters": [
            {"key": "alpha", "label": "正则强度", "type": "number", "default": .01, "min": 0},
            {"key": "l1_ratio", "label": "L1 比例", "type": "number", "default": .5, "min": 0, "max": 1},
        ],
    },
    "xgboost": {
        "name": "XGBoost", "family": "非线性模型", "formula": "y_hat = sum_k f_k(x), f_k in regression trees",
        "method": "直方图树提升拟合收益，Optuna 以收益与 IC 联合验证目标选参。",
        "parameters": [
            {"key": "max_depth", "label": "树深度", "type": "integer", "default": 4, "min": 1, "max": 10},
            {"key": "learning_rate", "label": "学习率", "type": "number", "default": .05, "min": .001, "max": .5},
            {"key": "n_estimators", "label": "最大树数", "type": "integer", "default": 200, "min": 20},
        ],
    },
    "style_rotation": {
        "name": "PCA-XGBoost", "family": "降维非线性模型",
        "formula": "Z -> PCA exposures -> XGBoost(next return)",
        "method": "PCA 提取正交风格暴露，再用 XGBoost 拟合股票收益；PCA 可滚动重估或固定。",
        "parameters": [
            {"key": "pca_variance", "label": "解释率", "type": "number", "default": .90, "min": .5, "max": .999},
            {"key": "pca_fit_mode", "label": "PCA 模式", "type": "select", "default": "rolling", "options": ["rolling", "fixed"]},
            {"key": "holding_count", "label": "持仓只数", "type": "integer", "default": 200, "min": 1},
        ],
    },
}


def framework_manifest() -> dict[str, Any]:
    return {
        "architecture": [
            {"id": "data", "name": "数据层", "description": "自动识别本地 Parquet 与 Tushare API，统一日期和股票代码。"},
            {"id": "factor", "name": "因子层", "description": "FactorRegistry 注册 compute()，缺失告警跳过，逐日截面 Z-Score。"},
            {"id": "factor_backtest", "name": "因子回测层", "description": "IC、分层、多空和持有期稳定性分析。"},
            {"id": "model_backtest", "name": "模型回测层", "description": "滚动训练、组合构建、成本与约束统一执行。"},
            {"id": "visualization", "name": "可视化层", "description": "研究图表只读取标准结果，不耦合计算实现。"},
        ],
        "models": MODEL_CATALOG,
        "ic_horizons": [5, 10, 20, 60],
        "rebalance_frequencies": ["D", "W", "M", "Q"],
        "weight_modes": ["equal", "score"],
        "optimizers": ["turnover", "max_drawdown", "transaction_cost", "industry_neutral"],
        "metrics": ["annual_return", "annual_volatility", "sharpe_rf_2pct", "max_drawdown",
                    "calmar", "win_rate", "profit_loss_ratio", "underwater_ratio"],
    }
