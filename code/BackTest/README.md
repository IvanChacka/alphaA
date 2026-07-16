# BackTest：Top200 因子/模型回测框架

本框架在交易日 `T` 使用 `T-1` 收盘后得到的因子或模型预测值进行选股。首次等权买入 Top200，之后每天最多替换30%持仓，并处理 TWAP 成交、14bp 双边费用、ST、停牌、涨跌停、交易单位及复权导致的持股数量变化。

## 启动网页

在 `code/BackTest` 目录执行：

```powershell
pip install -r requirements.text
python web_app.py
```

浏览器访问 `http://127.0.0.1:8000`。网页可以选择：

- 因子或模型
- A500、中证1000或全市场票池
- 开始日期、结束日期
- 初始资金

回测过程中，左侧实时显示交易记录；右侧实时显示收盘持仓、总资产和累计收益。完成后显示净值曲线、超额收益指标和完整后台明细。

## 命令行运行

```powershell
python main.py --signal default_factor --market A500 --start 2020-01-01 --end 2024-12-31
```

使用 `python main.py -h` 查看当前可选择的因子、模型和票池。

## 扩展因子/模型接口

内置 `default_factor`，读取项目 `BackTestData/factordata.parquet`。

模型检测接口为：

```text
GET /api/options
```

后端会自动扫描项目根目录的 `signal/`。增加模型时，新建模型目录并放入预测 CSV：

```text
signal/
└─ lr_baseline_252_1/
   ├─ 20220104.csv
   └─ 20220401.csv
```

每个 CSV 至少包含：

```text
date,ticker,mean
20220104,000001,0.61
```

其中 `date` 是交易日，`ticker` 是股票代码，最后一个非键字段作为预测值。刷新网页后，新模型会自动出现在“因子/模型”下拉框中。

## 输出命名

所有结果写入 `code/BackTest/output/`，命名规则统一为：

```text
因子或模型名_池子名_文件名
```

例如选择 `default_factor` 和 `A500`：

```text
default_factor_A500_account_daily.csv
default_factor_A500_holdings_daily.csv
default_factor_A500_orders.csv
default_factor_A500_closed_trades.csv
default_factor_A500_metrics.json
default_factor_A500_index.html
```

不同模型或票池的结果不会互相覆盖。

## 主要模块

- `config.py`：回测参数、票池和因子/模型发现接口
- `data_loader.py`：行情、票池、因子和模型预测数据加载
- `backtest.py`：交易与持仓逻辑
- `analytics.py`：收益、夏普、波动率和最大回撤
- `report.py`：结果文件和静态报告
- `web_app.py`：本地网页服务及实时状态接口
- `web/index.html`：网页控制台

## 重要计算口径

- `T` 日交易只能使用 `T-1` 日因子或模型值。
- TWAP 和收盘价视为后复权价，实际价格为后复权价除以当日累计复权因子。
- 300、688开头股票的涨跌停比例为20%，其他股票为10%。
- 普通股票交易单位为100股，688开头股票为200股。
- 无法卖出的涨跌停或停牌持仓继续持有。
- 复权因子变化时，持股数量乘以新旧复权因子之比，单位成本反向调整。
