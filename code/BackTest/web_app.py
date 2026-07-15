"""本地回测网页服务：提交参数、后台运行、查询进度和查看报告。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from email import policy
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
from urllib.parse import parse_qs, urlparse

from backtest import BacktestCancelled, BacktestEngine
from config import BacktestConfig, MARKETS, ROOT, available_signals
from data_loader import DataError, MarketData
from report import artifact_group, artifact_prefix, export_result

WEB_DIR = ROOT / "web"


class BacktestJob:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = {"status": "idle", "progress": 0, "message": "等待启动"}
        self.result = None
        self.live_trades = []
        self.live_holdings = []
        self.cancel_event = threading.Event()

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def start(self, params):
        with self.lock:
            if self.state["status"] in {"loading", "running", "stopping"}:
                return False
            self.state = {"status": "loading", "progress": 1, "message": "正在加载数据…"}
            self.live_trades = []
            self.live_holdings = []
            self.cancel_event.clear()
        threading.Thread(target=self._run, args=(params,), daemon=True).start()
        return True

    def stop(self):
        with self.lock:
            if self.state["status"] not in {"loading", "running", "stopping"}:
                return False
            self.cancel_event.set()
            self.state.update(status="stopping", message="正在安全终止回测…")
        return True

    def _run(self, params):
        try:
            cfg = replace(
                BacktestConfig(),
                market=params["market"],
                signal=params["signal"],
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
                # 更新实时数据在锁外进行（仅回测线程写入，无竞争）
                self.live_trades.extend(trades)
                self.live_trades = self.live_trades[-500:]
                self.live_holdings = holdings
                # 仅在锁内做原子赋值，最小化锁持有时间
                snapshot_trades = list(self.live_trades)
                snapshot_holdings = list(self.live_holdings)
                with self.lock:
                    self.state.update(
                        current_date=date.strftime("%Y-%m-%d"), total_asset=total_asset,
                        cumulative_return=total_asset / initial_cash - 1,
                        live_trades=snapshot_trades, live_holdings=snapshot_holdings,
                    )

            result = BacktestEngine(
                data, cfg, progress, daily, self.cancel_event.is_set
            ).run()
            self.result = result
            self.update(status="running", progress=97, message="正在生成报告…")
            group = artifact_group(cfg.signal, cfg.market)
            run_output = cfg.output_dir / group
            metrics = export_result(
                result, run_output, cfg.annual_days, cfg.risk_free_rate,
                cfg.signal, cfg.market,
            )
            report_name = artifact_prefix(cfg.signal, cfg.market) + "index.html"
            self.update(
                status="completed", progress=100, message="回测完成",
                metrics=metrics, report=f"/report/{group}/{report_name}",
            )
        except BacktestCancelled:
            self.update(status="stopped", message="回测已终止", progress=0)
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
        if path == "/api/options":
            return self._json({"markets": list(MARKETS), "signals": available_signals()})
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
        request_path = urlparse(self.path).path
        if request_path == "/api/stop":
            if JOB.stop():
                return self._json({"ok": True}, 202)
            return self._json({"error": "当前没有正在运行的回测"}, 409)
        if request_path == "/api/upload":
            return self._upload_signal()
        if request_path != "/api/start":
            return self.send_error(404)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            params = json.loads(self.rfile.read(length))
            if params.get("market") not in MARKETS:
                raise ValueError("不支持的票池")
            if params.get("signal") not in available_signals():
                raise ValueError("不支持的因子/模型")
            if float(params.get("cash", 0)) <= 0:
                raise ValueError("初始资金必须大于 0")
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._json({"error": str(exc)}, 400)
        if not JOB.start(params):
            return self._json({"error": "已有回测正在运行"}, 409)
        self._json({"ok": True}, 202)

    def _upload_signal(self):
        """接收 multipart CSV/Parquet，验证后保存到 BackTestData。"""
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 256 * 1024 * 1024:
                raise ValueError("上传文件必须小于256MB")
            content_type = self.headers.get("Content-Type", "")
            if not content_type.startswith("multipart/form-data"):
                raise ValueError("上传请求必须使用 multipart/form-data")
            envelope = (
                f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
                + self.rfile.read(length)
            )
            message = BytesParser(policy=policy.default).parsebytes(envelope)
            part = next(
                (item for item in message.iter_parts() if item.get_filename()), None
            )
            if part is None:
                raise ValueError("没有收到文件")
            original = Path(part.get_filename()).name
            suffix = Path(original).suffix.lower()
            if suffix not in {".csv", ".parquet"}:
                raise ValueError("只支持 CSV 或 Parquet 文件")
            stem = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]+", "_", Path(original).stem).strip("_")
            if not stem:
                raise ValueError("文件名无效")
            filename = stem + suffix
            upload_dir = BacktestConfig().data_dir / "uploaded_signals"
            upload_dir.mkdir(parents=True, exist_ok=True)
            target = upload_dir / filename
            target.write_bytes(part.get_payload(decode=True))
            try:
                MarketData._read_uploaded_signal(target)
            except Exception:
                target.unlink(missing_ok=True)
                raise
            key = f"upload:{filename}"
            self._json({"ok": True, "signal": key, "filename": filename,
                        "signals": available_signals()}, 201)
        except (ValueError, DataError, StopIteration) as exc:
            self._json({"error": str(exc)}, 400)

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
