# alphaA

面向A股的因子研究、滚动机器学习、PCA风格择时和Top 200回测工程。项目将模型训练与交易执行分离：模型只生成每日股票预测分数和风格投票，所有交易统一进入既有BackTest框架，保留TWAP、涨跌停、ST、停牌、交易单位、复权和费用约束。

> 当前研究结果仅覆盖2024年样本外区间，不构成实盘或投资建议。继续调参前应冻结当前配置，并使用独立年份验证。

## 项目入口

- [回测框架说明](code/BackTest/README.md)
- [滚动模型与PCA风格模型说明](code/Model/README.md)
- [最新模型对比报告](code/Model/output/model_comparison_20260716/index.html)
- [最新详细分析](code/Model/output/model_comparison_20260716/analysis.md)
- [策略调整全过程](code/Model/output/model_comparison_20260716/strategy_evolution.md)

## 目录结构

```text
alphaA/
├─ BackTestData/                 # 本地行情、因子、票池和基准数据
├─ code/
│  ├─ BackTest/                 # 统一交易与账户回测框架
│  │  ├─ config.py              # 回测参数、票池和信号发现
│  │  ├─ data_loader.py         # 行情、票池、因子/模型读取
│  │  ├─ backtest.py            # 交易、持仓、复权和账户逻辑
│  │  ├─ analytics.py           # 绝对与相对净值指标
│  │  ├─ main.py                # 回测命令行入口
│  │  ├─ web_app.py             # 回测实时网页服务
│  │  └─ output/                # 独立回测输出
│  └─ Model/
│     ├─ env.py                 # 模型、路径和优化器统一参数
│     ├─ main_rolling.py        # Linear/Ridge/ElasticNet/XGBoost入口
│     ├─ main_style_rotation.py # 固定PCA风格轮动入口
│     ├─ web_app.py             # 模型训练实时网页服务
│     ├─ rolling_ml/            # 滚动数据、训练、指标、报告和适配器
│     ├─ style_rotation/        # PCA、风格收益、双期限择时和残差模型
│     ├─ optimizers/            # 行业中性、惰性换手、QP动态仓位
│     ├─ tests/                 # 对齐、防泄漏、回测和网页测试
│     └─ output/run_*/          # 每次模型运行的完整产物
└─ README.md
```

`code/Model`是模型训练总入口，完成预测后通过`BacktestAdapter`调用`code/BackTest`，不会复制或绕过回测交易规则。

## 环境安装

建议使用64位Python 3.10以上版本。当前工程已在Python 3.13环境运行。

```powershell
cd path\to\alphaA
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r code\Model\requirements.txt
```

模型依赖包括：`pandas、numpy、pyarrow、scikit-learn、xgboost、optuna、plotly、scipy、joblib、pytest`。BackTest单独运行时也可安装：

```powershell
pip install -r code\BackTest\requirements.text
```

## 数据文件

程序只读使用项目根目录的`BackTestData/`，不会修改原始数据：

```text
BackTestData/
├─ factordata.parquet        # ticker,date索引/字段；数值列为因子
├─ twap.parquet              # 后复权TWAP宽表
├─ closePrice.parquet        # 后复权收盘价宽表
├─ accumAdjFactor.parquet    # 累计后复权因子
├─ ST.parquet                # ST状态
├─ pool_A500.parquet         # A500历史成分池
├─ pool_zz1000.parquet       # ZZ1000历史成分池
├─ Benchmark_A500.parquet    # A500基准
├─ Benchmark_zz1000.parquet  # ZZ1000基准
├─ calendar.csv              # calendarDate、isOpen交易日历
├─ industry.csv/parquet      # 可选，行业中性化使用
└─ uploaded_signals/         # 网页上传的外部因子/模型文件
```

股票代码统一为六位字符串。Parquet文件必须是真实Parquet格式，不能仅将CSV扩展名改为`.parquet`。

大体积数据和单次运行生成的`output/run_*`默认不提交Git；克隆项目后需要自行放回本地数据。仓库只保留经过脱敏的汇总分析文件，不包含逐日预测、订单、持仓或完整运行日志。

## 快速启动模型网页

```powershell
cd path\to\alphaA\code\Model
python web_app.py
```

访问：<http://127.0.0.1:8090>

网页支持：

- 多选Linear、Ridge、ElasticNet、XGBoost和固定PCA风格轮动模型。
- 后台按模型顺序运行，保留已完成面板。
- 实时查看训练阶段、参数/验证折进度、RankIC和回测净值。
- 选择`A500、ZZ1000、ALL_A500、ALL_ZZ1000`。
- 配置PCA解释率、残差权重、风格投票和滚动ICIR窗口。
- 选择行业中性、惰性换手和二次规划回撤优化器。
- 随时终止当前训练子进程。

端口受限时可更换端口：

```powershell
python web_app.py --host 127.0.0.1 --port 8091
```

## 当前推荐研究策略

根据2024年现有报告，当前采用稳健性优先的主策略：

```text
固定PCA90风格
+ 5日/20日XGBoost择时
+ 15个已实现IC的滚动ICIR
+ residual_weight=0
+ lazy_turnover
+ quadratic_drawdown
```

命令行完整启动：

```powershell
cd path\to\alphaA\code\Model
python main_style_rotation.py `
  --test-year 2024 `
  --pools ALL_ZZ1000 `
  --style-models xgboost_dual_horizon `
  --pca-variance 0.90 `
  --pca-min-components 5 `
  --buy-confirmations 1 `
  --sell-confirmations 2 `
  --vote-quantile 0.70 `
  --residual-weight 0 `
  --icir-window 15 `
  --optimizer none `
  --turnover-optimizer lazy_turnover `
  --max-turnover-ratio 0.30 `
  --drawdown-optimizer quadratic_drawdown `
  --max-drawdown-limit 0.20 `
  --n-jobs 12
```

该次配置对应`run_20260716_165342_style_rotation`：

| 指标 | 2024样本外结果 |
|---|---:|
| 年化收益 | 15.16% |
| 累计收益 | 14.46% |
| 夏普 | 0.677 |
| 最大回撤 | -17.06% |
| 相对ZZ1000年化超额 | 13.08% |
| 超额夏普 | 0.751 |
| RankIC / ICIR | 0.0396 / 0.3967 |
| 换手 / 交易费 | 15.82 / 209.84万元 |
| 平均 / 最低目标仓位 | 59.05% / 30% |

选择它不是因为每项收益最高，而是因为相对未优化风格策略，最大回撤约减半，换手和费用下降约75%，同时避免残差alpha和权重对单一年份的过拟合。

## 运行基础滚动模型

完整基础模型：

```powershell
cd path\to\alphaA\code\Model
python main_rolling.py --test-year 2024 --pools A500 ZZ1000 --models linear_regression ridge elasticnet xgboost --optuna-trials 30 --n-jobs 12
```

快速验证Linear且跳过回测：

```powershell
python main_rolling.py --models linear_regression --pools A500 --optuna-trials 1 --skip-backtest
```

固定四年滚动窗口：

```powershell
python main_rolling.py --window-mode fixed
```

## 单独启动BackTest

网页启动：

```powershell
cd path\to\alphaA\code\BackTest
python web_app.py
```

访问：<http://127.0.0.1:8000>

命令行示例：

```powershell
python main.py --signal default_factor --market A500 --start 2020-01-01 --end 2024-12-31 --cash 100000000
```

使用`python main.py -h`查看当前可用票池和自动发现的因子/模型。

## 票池含义

| 选项 | 选股范围 | 基准 |
|---|---|---|
| `A500` | A500历史成分股 | A500 |
| `ZZ1000` | ZZ1000历史成分股 | ZZ1000 |
| `ALL_A500` | 因子文件中的全部可用股票 | A500 |
| `ALL_ZZ1000` | 因子文件中的全部可用股票 | ZZ1000 |

`ALL_ZZ1000`不是中证1000成分池，而是全市场选股、以ZZ1000作为业绩基准。

## 时间与防未来函数口径

```text
t0：因子/模型预测在收盘后可获得
t1：使用t0预测，以真实TWAP交易
t2：标签退出价格
```

股票标签为：

```python
label_t0 = real_twap_t2 / real_twap_t1 - 1
real_twap = adjusted_twap / accum_adj_factor
```

训练样本是否可用由`label_exit_date`判断，而不是只看因子日期。PCA只使用2020—2021历史数据拟合一次；2022以后永久冻结载荷。20日隔离验证只在2024开始前选择树数，测试期不使用未来季度调参。

## 缺失值处理

处理顺序统一为：

1. 删除当天所有因子都缺失的股票行。
2. 保留部分因子缺失。
3. 使用训练期或当日截面的非缺失观测完成标准化。
4. 最后将剩余缺失值填0，表示标准化后的中性暴露。

不生成缺失指示变量，不执行MAD或中位数填充；发现`Inf`立即报错。

## 回测交易规则

- 第一个可交易日根据上一日因子买入Top 200，后续目标持有200只。
- 股票内部等权，名称换手每日最多30%。
- 交易使用给定TWAP，双边费率14bp。
- ST股票不能买入；交易价格缺失视为停牌，当日不能交易。
- 300、688开头股票涨跌停阈值20%，其他股票10%。
- 涨跌停股票不能买入或卖出。
- 普通股票交易数量为100股整数倍，688股票为200股整数倍。
- 累计复权因子变化时，持股数量按新旧复权因子比调整。
- `lazy_turnover`只决定已有持仓是否优先保留。
- `quadratic_drawdown`只调整30%—100%的组合目标总仓位，不能保证实际最大回撤绝对最小。

## 输出说明

每次模型运行生成独立目录：

```text
code/Model/output/run_<时间>_<模型>/
├─ config/run_manifest.json       # 本次全部参数
├─ logs/live_status.json          # 实时状态
├─ predictions/                   # 每日股票预测
├─ metrics/                       # RankIC、ICIR、PCA逐风格指标
├─ style/                         # PCA载荷、解释率、风格收益
├─ timing/                        # 动态权重、投票、特征重要性
├─ audit/                         # 日期、标签和数据质量审计
├─ backtest/                      # 账户、订单、持仓、费用和风险事件
└─ report/index.html              # 无CDN依赖的静态报告
```

关键文件：

- `metrics/daily_rankic.csv`：最终股票分数每日RankIC。
- `metrics/pca_style_daily_rankic.csv`：每个PCA风格逐日RankIC。
- `metrics/pca_style_rankic.csv`：PCA逐风格季度汇总。
- `metrics/score_component_rankic.csv`：风格、残差和最终分数组件IC。
- `backtest/results/summary.csv`：各票池回测汇总。
- `backtest/results/<模型_票池>/nav_curve.csv`：策略、基准和相对净值。
- `backtest/audit/*_risk_events.csv`：QP目标仓位和风险状态。

旧静态报告不会因代码口径修正自动更新，需要重新运行或重新生成报告。

## 指标口径

- RankIC：每日股票横截面的Spearman相关。
- ICIR：`Mean(Daily RankIC) / Std(Daily RankIC)`，不乘`√252`。
- 绝对年化：由策略净值几何复利计算。
- 超额净值：`策略净值 / 基准净值`。
- 超额年化、超额夏普、超额波动和超额回撤：全部由相对净值收益计算。
- 换手：成交总金额除以回测期平均资产，不是百分数展示。

高RankIC不保证扣费后收益更高，必须联合比较回撤、换手、费用和目标仓位。

## 测试

```powershell
cd path\to\alphaA\code\Model
python -m pytest tests -q
```

当前测试覆盖数据读取、缺失值顺序、时间切分、标签对齐、未来函数、常数预测、PCA风格、优化器、停牌估值、绩效口径和网页接口。

## 常见问题

### `Parquet magic bytes not found`

文件不是有效Parquet或已损坏。不要只修改扩展名；重新导出真实Parquet文件。

### `Setting a MultiIndex dtype ... is not supported`

通常来自旧版读取代码直接修改MultiIndex类型。当前适配器会先`reset_index()`，再分别规范`ticker`和`date`。修改代码后需要重启网页后端。

### `[WinError 10013]`或端口无法绑定

先使用`127.0.0.1`并换一个端口：

```powershell
python web_app.py --host 127.0.0.1 --port 8091
```

### 网页显示连接断开

确认后台Python进程仍在、端口仍监听，并检查运行目录的`logs/live_status.json`和终端日志。长任务由子进程顺序运行，避免同时启动多个模型争抢内存。

### 训练很慢

- 一次只选择一个复杂模型。
- XGBoost使用`hist`和12线程；Optuna Trial顺序执行，避免嵌套并行。
- ElasticNet坐标下降不一定占满全部CPU。
- 查看网页当前参数组合、验证折和季度进度，先确认是在计算而不是退出。

### 终止按钮

网页终止按钮会结束当前训练子进程。已生成的运行目录和日志保留，可用于定位停止位置。

### 输出目录为空

先查看`logs/live_status.json`和`logs/failed_tasks.csv`。只有任务完成相应阶段后才会生成预测、指标和报告文件。

## 当前限制

- 当前策略选择主要基于2024年，尚未完成跨年度、跨周期独立验证。
- `ALL_*`全市场结果可能包含行业和市值暴露；行业中性化需单独测试。
- 回测已计14bp交易费，但未完整模拟冲击成本和组合容量。
- QP优化的是最大回撤风险代理，不是对未来真实最大回撤的数学保证。

后续应冻结当前参数，优先进行多年度滚动验证、不同票池迁移、28bp费用压力测试和成交容量分析。
