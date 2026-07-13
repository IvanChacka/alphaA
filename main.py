import argparse
from dataclasses import replace
from config import BacktestConfig, MARKETS
from data_loader import DataError, MarketData
from backtest import BacktestEngine
from report import export_result


def main():
    p = argparse.ArgumentParser(description="因子 Top200 回测")
    p.add_argument("--market", choices=MARKETS, default="A500")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--cash", type=float, default=100_000_000)
    args = p.parse_args()
    cfg = replace(BacktestConfig(), market=args.market, start_date=args.start, end_date=args.end, initial_cash=args.cash)
    try:
        data = MarketData(cfg).load()
    except DataError as exc:
        p.error(str(exc))
    result = BacktestEngine(data, cfg).run()
    metrics = export_result(result, cfg.output_dir, cfg.annual_days, cfg.risk_free_rate)
    print(f"完成：{cfg.output_dir / 'index.html'}")
    print(metrics)


if __name__ == "__main__": main()
