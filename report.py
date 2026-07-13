"""保存后台记录并生成可独立打开的中文静态报告。"""
from pathlib import Path
import html
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from analytics import performance


METRIC_LABELS = {
    "absolute_annual_return": "绝对年化收益率", "absolute_sharpe": "绝对收益夏普比率",
    "absolute_annual_volatility": "绝对年化波动率", "absolute_max_drawdown": "绝对最大回撤",
    "excess_annual_return": "超额年化收益率", "excess_sharpe": "超额收益夏普比率",
    "excess_annual_volatility": "超额年化波动率", "excess_max_drawdown": "超额最大回撤",
}


def _display_table(frame: pd.DataFrame, limit=200):
    data = frame.tail(limit).copy()
    if isinstance(data.index, pd.DatetimeIndex):
        data.index = data.index.strftime("%Y-%m-%d")
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].dt.strftime("%Y-%m-%d")
    return data.to_html(classes="data-table", border=0, na_rep="", index=True)


def export_result(result, output: Path, annual_days=252, rf=0.0):
    output.mkdir(parents=True, exist_ok=True)
    metrics, curve = performance(result.account, result.benchmark, annual_days, rf)

    # 完整后台记录。网页预览不截断 CSV 文件。
    result.account.to_csv(output / "account_daily.csv", encoding="utf-8-sig")
    result.holdings.to_csv(output / "holdings_daily.csv", index=False, encoding="utf-8-sig")
    result.orders.to_csv(output / "orders.csv", index=False, encoding="utf-8-sig")
    result.trades.to_csv(output / "closed_trades.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(output / "nav_curve.csv", encoding="utf-8-sig")
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig = go.Figure()
    for col, label in [("strategy", "策略净值"), ("benchmark", "基准净值"), ("excess", "超额净值")]:
        fig.add_scatter(x=curve.index, y=curve[col], name=label)
    fig.update_layout(template="plotly_white", height=480, margin=dict(l=40, r=20, t=40, b=40),
                      hovermode="x unified", legend=dict(orientation="h"))

    cards = "".join(
        f'<div class="card"><span>{html.escape(METRIC_LABELS.get(k, k))}</span>'
        f'<b>{v:.3f}</b></div>' if "sharpe" in k else
        f'<div class="card"><span>{html.escape(METRIC_LABELS.get(k, k))}</span>'
        f'<b>{v:.2%}</b></div>'
        for k, v in metrics.items()
    )
    downloads = "".join([
        '<a href="account_daily.csv">每日账户 CSV</a>', '<a href="holdings_daily.csv">每日持仓 CSV</a>',
        '<a href="orders.csv">全部委托 CSV</a>', '<a href="closed_trades.csv">平仓交易 CSV</a>',
        '<a href="nav_curve.csv">净值曲线 CSV</a>',
    ])
    sections = [
        ("每日账户记录（最近200条）", result.account),
        ("每日收盘持仓（最近200条）", result.holdings),
        ("全部委托记录（最近200条）", result.orders),
        ("完整买卖与盈亏记录（最近200条）", result.trades),
    ]
    tables = "".join(
        f'<section><h2>{title}</h2><div class="table-wrap">{_display_table(frame)}</div></section>'
        for title, frame in sections
    )
    page = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>多因子回测完整报告</title>
<style>body{{font-family:Inter,"Microsoft YaHei",sans-serif;margin:0;background:#f4f7fb;color:#18212f}}
.wrap{{max-width:1400px;margin:30px auto;padding:0 20px}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}
.card,section{{background:white;border:1px solid #e4e9f0;border-radius:12px;padding:18px;margin-bottom:18px}}
.card span{{display:block;color:#687588;font-size:13px;margin-bottom:7px}}.card b{{font-size:23px}}.downloads{{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}}
.downloads a{{color:white;background:#1769e0;text-decoration:none;padding:9px 13px;border-radius:7px}}.table-wrap{{overflow:auto;max-height:520px}}
table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px 10px;border-bottom:1px solid #e8edf3;text-align:right;white-space:nowrap}}th{{position:sticky;top:0;background:#eef3f9}}h2{{font-size:18px}}
@media(max-width:800px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}</style></head><body><main class="wrap">
<h1>多因子回测完整报告</h1><div class="cards">{cards}</div><div class="downloads">{downloads}</div>
<section><h2>策略、基准与超额净值曲线</h2>{fig.to_html(full_html=False, include_plotlyjs=True)}</section>{tables}
</main></body></html>'''
    (output / "index.html").write_text(page, encoding="utf-8")
    return metrics
