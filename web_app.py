"""本地回测网页服务：提交参数、后台运行、查询进度和查看报告。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent
BACKTEST_DIR = PROJECT_ROOT / "code" / "BackTest"
if BACKTEST_DIR.exists():
    BACKTEST_DIR_STR = str(BACKTEST_DIR)
    if BACKTEST_DIR_STR not in sys.path:
        sys.path.insert(0, BACKTEST_DIR_STR)

from backtest import BacktestEngine
from config import BacktestConfig, MARKETS, ROOT
from data_loader import DataError, MarketData
from report import export_result

WEB_DIR = ROOT / "web"


class BacktestJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"status": "idle", "progress": 0, "message": "等待启动"}
        self.result = None
        self.live_trades = []
        self.live_holdings = []

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def start(self, params):
        with self.lock:
            if self.state["status"] in {"loading", "running"}:
                return False
            self.state = {"status": "loading", "progress": 1, "message": "正在加载数据…"}
            self.live_trades = []
            self.live_holdings = []
        threading.Thread(target=self._run, args=(params,), daemon=True).start()
        return True

    def _run(self, params):
        try:
            cfg = replace(
                BacktestConfig(),
                market=params["market"],
                start_date=params.get("start") or None,
                end_date=params.get("end") or None,
                initial_cash=float(params.get("cash", 100_000_000)),
            )
            data = MarketData(cfg).load()

            def progress(done, total, date):
                self.update(
                    status="running",
                    progress=max(2, int(done / total * 96)),
                    message=f"回测中：{date:%Y-%m-%d}（{done}/{total}）",
                )

            def daily(date, trades, holdings, total_asset, initial_cash):
                with self.lock:
                    self.live_trades.extend(trades)
                    self.live_trades = self.live_trades[-500:]
                    self.live_holdings = holdings
                    self.state.update(
                        current_date=date.strftime("%Y-%m-%d"), total_asset=total_asset,
                        cumulative_return=total_asset / initial_cash - 1,
                        live_trades=list(self.live_trades), live_holdings=list(self.live_holdings),
                    )

            result = BacktestEngine(data, cfg, progress, daily).run()
            self.result = result
            self.update(status="running", progress=97, message="正在生成报告…")
            metrics = export_result(
                result, cfg.output_dir, cfg.annual_days, cfg.risk_free_rate
            )
            self.update(
                status="completed", progress=100, message="回测完成",
                metrics=metrics, report="/report/index.html",
            )
        except (DataError, ValueError, KeyError) as exc:
            self.update(status="failed", progress=0, message=str(exc))
        except Exception as exc:
            self.update(status="failed", progress=0, message=f"运行失败：{exc}")


JOB = BacktestJob()


class Handler(SimpleHTTPRequestHandler):
    def _json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, content_type):
        if not path.is_file():
            return self.send_error(404)
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/status":
            return self._json(JOB.snapshot())
        if path == "/api/table":
            if JOB.result is None:
                return self._json({"error": "尚无回测结果"}, 404)
            query = parse_qs(parsed.query)
            name = query.get("name", [""])[0]
            tables = {
                "account": JOB.result.account.reset_index(),
                "holdings": JOB.result.holdings,
                "orders": JOB.result.orders,
                "trades": JOB.result.trades,
            }
            if name not in tables:
                return self._json({"error": "未知数据表"}, 400)
            frame = tables[name]
            page = max(1, int(query.get("page", ["1"])[0]))
            size = min(200, max(10, int(query.get("size", ["50"])[0])))
            start = (page - 1) * size
            records = frame.iloc[start:start + size].copy()
            for column in records.columns:
                if str(records[column].dtype).startswith("datetime"):
                    records[column] = records[column].dt.strftime("%Y-%m-%d")
            rows = json.loads(records.to_json(orient="records", date_format="iso"))
            return self._json({"columns": list(records.columns), "records": rows,
                               "page": page, "pages": max(1, (len(frame) + size - 1) // size), "total": len(frame)})
        if path in {"/", "/index.html"}:
            return self._file(WEB_DIR / "index.html", "text/html; charset=utf-8")
        if path.startswith("/report/"):
            target = (ROOT / "output" / path.removeprefix("/report/")).resolve()
            output = (ROOT / "output").resolve()
            if target != output and output not in target.parents:
                return self.send_error(403)
            kind = "text/html; charset=utf-8" if target.suffix == ".html" else "application/octet-stream"
            return self._file(target, kind)
        self.send_error(404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/start":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            params = json.loads(self.rfile.read(length))
            if params.get("market") not in MARKETS:
                raise ValueError("不支持的票池")
            if float(params.get("cash", 0)) <= 0:
                raise ValueError("初始资金必须大于 0")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._json({"error": str(exc)}, 400)
        if not JOB.start(params):
            return self._json({"error": "已有回测正在运行"}, 409)
        self._json({"ok": True}, 202)

    def log_message(self, fmt, *args):
        print(f"[Web] {fmt % args}")


def main():
    parser = argparse.ArgumentParser(description="启动本地回测网页")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"回测控制台：http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")


if __name__ == "__main__":
    main()
