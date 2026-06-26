#!/usr/bin/env python3
"""Compare multiple overnight exit modes on the same Top-N input table.

This script reuses the baseline overnight backtester but evaluates several
exit return columns side by side:
- open
- 09:35
- 09:45
- 10:00

It is intended as the first downstream analysis layer after multi-exit labels
become available.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from backtest_overnight_topn_baseline import EXIT_COLUMN_MAP, load_input, run_backtest


DEFAULT_INPUT = Path("data/overnight_mvp/backtest_inputs/topn_baseline_input_20260401_20260410.csv")
DEFAULT_OUTDIR = Path("data/overnight_mvp/multi_exit_comparison")
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_TOP_N = 5
DEFAULT_EXIT_MODES = ["open", "0935", "0945", "1000"]


def _fmt_pct_or_nan(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x:.4%}"


def build_markdown(summary_df: pd.DataFrame, input_path: Path) -> str:
    show = summary_df[[
        "exit_mode", "return_column", "available_rows", "coverage_ratio", "trade_days", "trade_count", "ending_capital",
        "total_return", "annualized_return", "max_drawdown", "sharpe", "calmar",
        "win_rate", "avg_trade_return_net", "status"
    ]].copy()
    if "coverage_ratio" in show.columns:
        show["coverage_ratio"] = show["coverage_ratio"].map(_fmt_pct_or_nan)
    for col in ["total_return", "annualized_return", "max_drawdown", "win_rate", "avg_trade_return_net"]:
        if col in show.columns:
            show[col] = show[col].map(_fmt_pct_or_nan)
    for col in ["sharpe", "calmar"]:
        if col in show.columns:
            show[col] = show[col].map(lambda x: "nan" if pd.isna(x) else f"{x:.4f}")

    best_total = summary_df.sort_values("total_return", ascending=False, na_position="last").head(1)
    best_sharpe = summary_df.sort_values("sharpe", ascending=False, na_position="last").head(1)

    return f"""# Multi-Exit Overnight Comparison

- Input: `{input_path}`
- Exit modes evaluated: `{', '.join(summary_df['exit_mode'].astype(str).tolist())}`

## Best by total return
{best_total.to_markdown(index=False)}

## Best by Sharpe
{best_sharpe.to_markdown(index=False)}

## Full comparison
{show.to_markdown(index=False)}
""" + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple overnight exit modes on the same Top-N input table")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Top-N baseline input CSV")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--exit-modes", default=",".join(DEFAULT_EXIT_MODES), help="Comma-separated exit modes from EXIT_COLUMN_MAP")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    exit_modes = [x.strip() for x in str(args.exit_modes).split(",") if x.strip()]
    rows = []
    daily_outputs = {}
    trade_outputs = {}

    for exit_mode in exit_modes:
        if exit_mode not in EXIT_COLUMN_MAP:
            rows.append({
                "exit_mode": exit_mode,
                "return_column": None,
                "available_rows": 0,
                "coverage_ratio": 0.0,
                "status": "unsupported_exit_mode",
            })
            continue
        try:
            raw = pd.read_csv(input_path)
            return_col = EXIT_COLUMN_MAP[exit_mode]
            available_rows = int(pd.to_numeric(raw.get(return_col), errors="coerce").notna().sum()) if return_col in raw.columns else 0
            coverage_ratio = float(available_rows / len(raw)) if len(raw) else float("nan")
            df = load_input(input_path, args.top_n, exit_mode=exit_mode)
            daily, trades, summary = run_backtest(
                df=df,
                initial_capital=args.initial_capital,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                top_n=args.top_n,
                exit_mode=exit_mode,
            )
            rows.append({
                "exit_mode": exit_mode,
                "return_column": return_col,
                "available_rows": available_rows,
                "coverage_ratio": coverage_ratio,
                "status": "ok",
                **summary,
            })
            daily_outputs[exit_mode] = daily
            trade_outputs[exit_mode] = trades
        except Exception as exc:
            rows.append({
                "exit_mode": exit_mode,
                "return_column": EXIT_COLUMN_MAP.get(exit_mode),
                "available_rows": available_rows if 'available_rows' in locals() else None,
                "coverage_ratio": coverage_ratio if 'coverage_ratio' in locals() else None,
                "status": f"error:{type(exc).__name__}",
                "error": str(exc),
            })

    summary_df = pd.DataFrame(rows).sort_values(["total_return", "sharpe"], ascending=[False, False], na_position="last").reset_index(drop=True)

    stem = input_path.stem.replace("topn_baseline_input_", "") + f"_top{args.top_n}"
    summary_path = outdir / f"multi_exit_summary_{stem}.md"
    csv_path = outdir / f"multi_exit_summary_{stem}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    summary_path.write_text(build_markdown(summary_df, input_path), encoding="utf-8")

    for exit_mode, daily in daily_outputs.items():
        daily.to_csv(outdir / f"multi_exit_daily_{stem}_{exit_mode}.csv", index=False)
    for exit_mode, trades in trade_outputs.items():
        trades.to_csv(outdir / f"multi_exit_trades_{stem}_{exit_mode}.csv", index=False)

    print(f"Wrote comparison CSV: {csv_path}")
    print(f"Wrote comparison MD: {summary_path}")
    if not summary_df.empty:
        best = summary_df.iloc[0]
        print(
            "Best exit mode: "
            f"{best.get('exit_mode')} total_return={best.get('total_return')} sharpe={best.get('sharpe')} status={best.get('status')}"
        )


if __name__ == "__main__":
    main()
