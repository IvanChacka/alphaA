# 滚动机器学习预测工程

## 目标

本工程训练集2020-2023年，实现2024年严格样本外的季度滚动预测。支持 LinearRegression、Ridge、ElasticNet 和 XGBoost；XGBoost使用Optuna TPE在测试年度开始前调参一次，季度只重新训练，不使用2024回测结果选参数。

## 目录

```text
code/Model/
├─ imports.py              # 统一依赖
├─ env.py                  # 全部路径和参数
├─ main_rolling.py         # 一键入口
├─ rolling_predict.py      # 兼容入口
├─ optimizers/             # 回测前预测分数约束（当前支持行业中性）
├─ rolling_ml/             # 数据、标签、模型、调参、适配、报告
├─ tests/                  # 时间对齐与防泄漏测试
└─ output/run_*/           # 每次运行独立产物
```

## 数据与字段

- `factordata.parquet`：MultiIndex为 `ticker,date`，其余数值列为特征。
- `twap.parquet`：后复权TWAP宽表。
- `accumAdjFactor.parquet`：累计后复权因子宽表。
- `calendar.csv`：使用 `calendarDate` 且 `isOpen=True` 的交易日。

股票代码统一为六位字符串，日期统一为 `DatetimeIndex`。读取器检查重复样本、字段、日期交集和缺失率，不使用 `bfill`，不修改原文件。

## 真实价格与标签

真实TWAP：

```python
real_twap = adjusted_twap / accum_adj_factor
```

因子日在 `t0` 收盘后产生特征；`t1` 用TWAP买入；`t2` 用TWAP卖出。因此保存在 `t0` 的标签是：

```python
label_t0 = real_twap_t2 / real_twap_t1 - 1
```

即宽表的：

```python
label = real_twap.shift(-2) / real_twap.shift(-1) - 1
```

每条样本同时保存 `factor_date、label_entry_date、label_exit_date`。训练边界使用 `label_exit_date <= train_cutoff`，而不是仅检查因子日期。预测仍保存在t0，回测框架在t1使用，不进行第二次shift。

## 滚动方式

默认训练期为2020至2023年，测试期为2024年。2023年四个季度作为扩展窗口验证折，年度参数在2023年末锁定。2024Q1至Q4逐季重训，后续季度只加入标签已经实现的数据。

默认使用扩展窗口：

```python
ROLLING_WINDOW_MODE = "expanding"
```

也支持固定四年窗口：

```bash
python main_rolling.py --window-mode fixed
```

统一数据适配器会在所有模型之前将部分缺失的原因子直接填为代表中性暴露的0，并删除30个原始因子全部缺失的股票日样本；不生成缺失指示变量，最终保持30个特征。模型内部不执行MAD或中位数填充；LinearRegression、Ridge和ElasticNet使用StandardScaler，XGBoost保持原始数值，PCA风格模型执行每日截面标准化。Inf仍会立即报错。

## 模型

- LinearRegression：无正则线性基线。
- Ridge：alpha只在2020—2023时间序列验证中选择。
- ElasticNet：alpha和l1_ratio只在2020—2023时间序列验证中选择。
- XGBoost：`tree_method=hist`，默认伪Huber目标，固定随机种子，单个模型使用12线程。

Optuna使用 `TPESampler`、SQLite和 `load_if_exists=True`，可中断续跑。目标是四个验证季度合并后的平均日度RankIC。最佳折的 `best_iteration` 中位数成为2024年固定树数，测试季度不用于early stopping。

## 安装

```powershell
cd D:\Ivan\alphaA\code\Model
pip install -r requirements.txt
```

当前环境若提示缺少XGBoost：

```powershell
pip install xgboost
```

## 运行

完整运行：

```powershell
python main_rolling.py --test-year 2024 --pools A500 ZZ1000 --models linear_regression ridge elasticnet xgboost --optuna-trials 30 --n-jobs 12
```

启用行业中性分数优化：

```powershell
python main_rolling.py --models ridge --pools A500 --optimizer industry_neutral
python main_style_rotation.py --pools A500 --optimizer industry_neutral
```

快速验证线性基线且不回测：

```powershell
python main_rolling.py --models linear_regression --optuna-trials 1 --skip-backtest
```

测试：

```powershell
python -m pytest tests
```

## 输出

每次运行创建 `output/run_YYYYMMDD_HHMMSS/`，包含配置快照、数据与标签审计、Optuna SQLite和Trial、季度模型、预处理器、季度/年度预测、日度RankIC、训练记录、原回测框架结果、失败任务日志和自包含 `report/index.html`。

HTML嵌入Plotly JavaScript和数据，不启动Web服务、不依赖CDN。大表只展示配置行数，完整数据保存在CSV/Parquet。

## 指标解读

RankIC按每日股票横截面计算Spearman相关；同时报告Pearson IC、ICIR（每日RankIC均值/标准差）、RankIC正比例、t统计量及Top-Bottom收益。ICIR不再乘以√252，避免与一年样本的IC显著性t统计量混淆。高RankIC不保证扣费后高收益，必须结合回测换手和费用判断。2024结果只用于最终样本外评价，禁止反向选择模型、特征或参数。

## 常见问题

- `No module named xgboost`：安装requirements。
- 内存不足：先只运行一个模型；因子数据会按配置日期下推过滤。
- 没有预测：查看 `logs/failed_tasks.csv` 和训练日志。
- Optuna中断：用相同运行目录的SQLite恢复；默认研究设置了 `load_if_exists=True`。
- 日期泄漏测试失败：检查交易日历以及 `label_exit_date`，不要按自然日偏移或使用bfill。
- 回测失败：适配器只调用现有BackTest接口；先单独确认 `code/BackTest` 可运行。

模型计算默认限制为12线程。Optuna Trial仍顺序执行，每个Trial内部的XGBoost使用12线程，避免Trial并行与模型并行叠加；线性模型的BLAS/OpenMP同样限制为12线程。ElasticNet的坐标下降实现不能保证占满12核。

ElasticNet年度参数选择使用最近2个季度的时序验证，共8组参数（4个alpha × 2个l1_ratio）；网页实时显示当前参数组合及验证折进度。季度正式训练仍使用截止当季前可获得的全部训练数据。

网页端PCA风格轮动采用单一双期限策略：PCA不限制成分个数，默认提取到累计解释率达到90%；分别以XGBoost预测未来5日与20日逐风格收益，使用最近60个已实现标签的滚动ICIR动态加权。任意一个风格明确看多即可进入买入候选；已有持仓只有在至少两个风格明确看空时才进入卖出候选，中性状态优先继续持有，没有买入候选时保留现金。残差模型先逐日从已实现股票收益中回归剔除PCA风格暴露可解释部分，再用原始因子的PCA正交补空间预测剩余收益；最终股票分数为“风格择时分数 + 残差增量权重 × 正交残差收益预测分数”。残差模型同样按季度训练，且只使用退出日早于当季预测起点的已实现标签。该持有优先信号仅作为BackTest的可选输入，其他模型的回测行为不变。

启动实时网页：

```powershell
python web_app.py
```

选择“PCA风格轮动”时会自动打开参数窗口，可调整PCA解释率、买卖确认数、投票分位、残差权重与alpha、滚动ICIR窗口；参数会写入该次运行的 `config/run_manifest.json`。

### 行业中性优化器

网页“分数优化器”可选择“不使用优化器”或“行业中性化”。首次使用行业中性化前，上传CSV或Parquet行业分类文件；后端通过校验后将其保存为 `BackTestData/industry.csv` 或 `BackTestData/industry.parquet`。支持两种字段格式：

```text
# 静态行业
ticker,industry
000001,bank

# 带生效日的历史行业
date,ticker,industry
2023-01-03,000001,bank
```

股票列也可命名为 `symbol`/`code`，行业列也可命名为 `industry_code`/`sector`，日期列也可命名为 `factor_date`/`effective_date`。历史行业只向后延用已经生效的分类，禁止 `bfill`；所选票池内有分数的股票必须100%具有当日行业分类，否则该票池回测会明确失败并列出缺失示例。

当前约束口径是：每天在所选票池内执行 `中性分数 = 原预测分数 - 当日行业均值`，再统一恢复分数波动率。这样使每个行业的选择分数均值为0并保留行业内排序。它只改变进入既有BackTest的预测分数，不修改最终持仓权重，也不改等权、30%换手、交易单位、ST/停牌、涨跌停、复权和费率规则。逐日约束审计保存在 `backtest/audit/*_industry_neutral.csv`。

网页市场选择支持 `A500`、`ZZ1000`、`ALL_A500` 和 `ALL_ZZ1000`。后两者使用因子文件中的全部可用股票，分别以A500和ZZ1000指数作为基准，不代表对应指数成分股池。

为控制风格择时过拟合，2020—2021仅用于固定PCA，XGBoost择时训练从2022开始。风格XGB将每个日期展开为“决策日×目标风格”样本，使用一个共享模型混合所有风格的5日/20日动量、波动率、市场状态、目标风格自身特征与风格标识，直接以原始收益率为目标预测每个风格的未来收益；不再为每个风格分别训练输出尺度不一致的XGB，也不对目标收益做倍数放大。20个交易日隔离期只在2024测试开始前的固定参数选择中执行一次，用于early stopping选择5日和20日模型的树数；进入2024后固定这组树数，每季度使用截止当季前全部已实现标签重训，不再重新切验证集或丢弃20日样本。原始收益尺度下使用 `min_child_weight=1`、`reg_alpha=0`、`reg_lambda=1`，模型深度仍限制为2，并保留行列采样。混合特征重要性保存在 `timing/xgb_feature_importance.csv`。

前端和静态报告对训练、隔离验证、2024样本外统一使用“预测风格收益 vs 同期限实现风格收益”的逐日跨风格RankIC，5日与20日分开展示。固定隔离验证只在参数选择时计算一次，因此图上只显示一个验证点，不再复制成季度水平线。最终股票分数的每日RankIC依然单独展示，不再与风格收益IC连线比较。5日/20日动态权重使用已经实现标签的XGB逐日预测ICIR，而不是原始动量ICIR；标签退出日必须早于当前决策日。样本外观察不足10日时使用固定验证IC作为先验，达到10日后滚动ICIR不为正的期限权重归零；两个期限同时失效时不产生新买入信号，已有持仓仍由多风格看空规则决定是否卖出。

风格分数、正交残差分数和最终合成分数的组件IC统一在“最终存在预测分数的买入候选股票”上计算，确保三条曲线股票样本完全一致。全市场残差IC仍以 `orthogonal_residual_all_universe` 单独保存在 `metrics/score_component_rankic.csv` 供审计，但不与候选样本曲线直接比较。IC本身不会相加：代码先执行 `最终分数 = 风格分数 + residual_weight × 残差分数`，再对最终分数与实现收益计算一次RankIC。

每次风格运行额外生成 `audit/momentum_baseline_ic.csv` 和 `audit/style_forecast_alignment.csv`：前者记录原始5日/20日动量的逐日跨风格IC，后者逐日、逐风格记录XGB预测收益、同期限实现收益和标签退出日。负IC不会被事后翻转，可通过这两张表区分市场风格反转与日期对齐错误。
