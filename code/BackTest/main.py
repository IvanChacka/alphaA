import argparse
from dataclasses import replace
from config import BacktestConfig, MARKETS, available_signals
from data_loader import DataError, MarketData
from backtest import BacktestEngine
from report import artifact_group, artifact_prefix, export_result


def main():
    p = argparse.ArgumentParser(description="因子 Top200 回测")
    p.add_argument("--market", choices=MARKETS, default="A500")
    p.add_argument("--signal", choices=available_signals(), default="default_factor")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--cash", type=float, default=100_000_000)
    args = p.parse_args()
    cfg = replace(BacktestConfig(), market=args.market, signal=args.signal,
                  start_date=args.start, end_date=args.end, initial_cash=args.cash)
    try:
        data = MarketData(cfg).load()
    except DataError as exc:
        p.error(str(exc))
    result = BacktestEngine(data, cfg).run()
    run_output = cfg.output_dir / artifact_group(cfg.signal, cfg.market)
    metrics = export_result(result, run_output, cfg.annual_days, cfg.risk_free_rate,
                            cfg.signal, cfg.market)
    print(f"完成：{run_output / (artifact_prefix(cfg.signal, cfg.market) + 'index.html')}")
    print(metrics)


if __name__ == "__main__": main()
