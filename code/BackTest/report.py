from pathlib import Path
import json
import plotly.graph_objects as go
import re
from analytics import performance


def artifact_prefix(signal: str, market: str) -> str:
    clean = lambda value: re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", value).strip("_")
    return f"{clean(signal)}_{clean(market)}_"


def artifact_group(signal: str, market: str) -> str:
    """每个因子/模型与票池组合使用独立结果目录。"""
    if signal.startswith("factor:"):
        signal = signal.split(":", 1)[1]
    elif signal.startswith("upload:"):
        signal = Path(signal.split(":", 1)[1]).stem
    clean = lambda value: re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", value).strip("_")
    return f"{clean(signal)}_{clean(market)}"


def export_result(result, output: Path, annual_days=252, rf=0.0,
                  signal="default_factor", market="A500"):
    output.mkdir(parents=True, exist_ok=True)
    prefix = artifact_prefix(signal, market)
    result.account.to_csv(output / f"{prefix}account_daily.csv", encoding="utf-8-sig")
    result.holdings.to_csv(output / f"{prefix}holdings_daily.csv", index=False, encoding="utf-8-sig")
    result.orders.to_csv(output / f"{prefix}orders.csv", index=False, encoding="utf-8-sig")
    result.trades.to_csv(output / f"{prefix}closed_trades.csv", index=False, encoding="utf-8-sig")
    metrics, curve = performance(result.account, result.benchmark, annual_days, rf)
    (output / f"{prefix}metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    fig = go.Figure()
    for col, label in [("strategy", "策略净值"), ("benchmark", "基准净值"), ("excess", "超额净值")]:
        fig.add_scatter(x=curve.index, y=curve[col], name=label)
    cards = "".join(f"<div class='card'><b>{k}</b><br>{v:.2%}</div>" if "sharpe" not in k else f"<div class='card'><b>{k}</b><br>{v:.3f}</div>" for k,v in metrics.items())
    html = f"<meta charset='utf-8'><title>回测报告</title><style>body{{font-family:Arial;margin:30px;background:#f6f7fb}}.cards{{display:flex;flex-wrap:wrap;gap:12px}}.card{{background:white;padding:16px;min-width:180px;border-radius:8px}}</style><h1>多因子回测报告</h1><div class='cards'>{cards}</div>{fig.to_html(full_html=False, include_plotlyjs=True)}"
    (output / f"{prefix}index.html").write_text(html, encoding="utf-8")
    return metrics
