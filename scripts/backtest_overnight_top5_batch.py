#!/usr/bin/env python3
"""Batch runner for Top-5 overnight baseline backtests.

This script reuses the first-pass single-run backtest and sweeps multiple
cost / score / input settings, producing a comparison table.

Current version focuses on a practical first sweep:
- same input table
- fixed Top-N (default Top-5)
- multiple fee/slippage combinations
- optional multiple initial-capital settings

Outputs:
  data/overnight_mvp/backtest_results/batch/top5_batch_summary_<stem>.csv
  data/overnight_mvp/backtest_results/batch/top5_batch_summary_<stem>.md
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backtest_overnight_topn_baseline import load_input, run_backtest


DEFAULT_INPUT = Path("data/overnight_mvp/backtest_inputs/topn_baseline_input_20260401_20260410.csv")
DEFAULT_OUTDIR = Path("data/overnight_mvp/backtest_results/batch")
DEFAULT_TOP_N = 5
DEFAULT_INITIAL_CAPITALS = "1000000"
DEFAULT_FEE_BPS_LIST = "5,10,15"
DEFAULT_SLIPPAGE_BPS_LIST = "0,5,10"


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _fmt_pct_or_nan(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x:.4%}"


def build_markdown(summary_df: pd.DataFrame, input_path: Path, top_n: int) -> str:
    display_cols = [
        "run_id",
        "fee_bps",
        "slippage_bps",
        "initial_capital",
        "trade_days",
        "trade_count",
        "total_return",
        "annualized_return",
        "max_drawdown",
        "sharpe",
        "calmar",
        "win_rate",
        "avg_trade_return_net",
        "ending_capital",
    ]
    disp = summary_df[display_cols].copy()
    for col in ["total_return", "annualized_return", "max_drawdown", "win_rate", "avg_trade_return_net"]:
        disp[col] = disp[col].map(_fmt_pct_or_nan)
    for col in ["sharpe", "calmar"]:
        disp[col] = disp[col].map(lambda x: "nan" if pd.isna(x) else f"{x:.4f}")

    best_return = summary_df.sort_values("total_return", ascending=False).head(1)
    best_sharpe = summary_df.sort_values("sharpe", ascending=False).head(1)

    text = f"""# Top-{top_n} Overnight Batch Backtest Summary

- Input: `{input_path}`
- Runs: `{len(summary_df)}`
- Top-N: `{top_n}`

## Best by total return
{best_return.to_markdown(index=False)}

## Best by Sharpe
{best_sharpe.to_markdown(index=False)}

## Full comparison
{disp.to_markdown(index=False)}
"""
    return text + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-run Top-5 overnight baseline backtests")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Top-N baseline input CSV")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top-N positions per day")
    parser.add_argument("--initial-capitals", default=DEFAULT_INITIAL_CAPITALS, help="Comma-separated list")
    parser.add_argument("--fee-bps-list", default=DEFAULT_FEE_BPS_LIST, help="Comma-separated list")
    parser.add_argument("--slippage-bps-list", default=DEFAULT_SLIPPAGE_BPS_LIST, help="Comma-separated list")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    initial_capitals = _parse_float_list(args.initial_capitals)
    fee_list = _parse_float_list(args.fee_bps_list)
    slippage_list = _parse_float_list(args.slippage_bps_list)

    base_df = load_input(input_path, args.top_n)

    rows = []
    run_id = 0
    for initial_capital, fee_bps, slippage_bps in product(initial_capitals, fee_list, slippage_list):
        run_id += 1
        daily, trades, summary = run_backtest(
            df=base_df.copy(),
            initial_capital=initial_capital,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            top_n=args.top_n,
        )
        rows.append({
            "run_id": run_id,
            "initial_capital": initial_capital,
            "fee_bps": fee_bps,
            "slippage_bps": slippage_bps,
            **summary,
        })

    summary_df = pd.DataFrame(rows).sort_values(["total_return", "sharpe"], ascending=[False, False]).reset_index(drop=True)

    stem = input_path.stem.replace("topn_baseline_input_", "") + f"_top{args.top_n}"
    csv_path = outdir / f"top5_batch_summary_{stem}.csv"
    md_path = outdir / f"top5_batch_summary_{stem}.md"

    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(summary_df, input_path, args.top_n), encoding="utf-8")

    print(f"Wrote batch CSV: {csv_path} rows={len(summary_df)}")
    print(f"Wrote batch MD: {md_path}")
    if not summary_df.empty:
        best = summary_df.iloc[0]
        print(
            "Best run: "
            f"run_id={int(best['run_id'])} fee={best['fee_bps']} slip={best['slippage_bps']} "
            f"total_return={best['total_return']:.6f} sharpe={best['sharpe']:.4f}"
        )


if __name__ == "__main__":
    main()
