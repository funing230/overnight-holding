#!/usr/bin/env python3
"""First-pass Top-N overnight baseline backtest.

Input:
  data/overnight_mvp/backtest_inputs/topn_baseline_input_<start>_<end>.csv

Strategy:
  - Rebalance daily
  - Buy Top-N candidates at T close
  - Sell at T+1 open
  - Equal-weight within selected Top-N
  - Apply simple round-trip cost + slippage in bps

Outputs:
  data/overnight_mvp/backtest_results/topn_backtest_summary_<suffix>.md
  data/overnight_mvp/backtest_results/topn_backtest_daily_<suffix>.csv
  data/overnight_mvp/backtest_results/topn_backtest_trades_<suffix>.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/overnight_mvp/backtest_inputs/topn_baseline_input_20260401_20260410.csv")
DEFAULT_OUTDIR = Path("data/overnight_mvp/backtest_results")
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0
DEFAULT_TOP_N = 5
DEFAULT_EXIT_MODE = "open"
TRADING_DAYS_PER_YEAR = 252

EXIT_COLUMN_MAP = {
    "open": "overnight_return_open",
    "0935": "overnight_return_0935",
    "0945": "overnight_return_0945",
    "1000": "overnight_return_1000",
}


def _fmt_pct(x: float) -> str:
    return f"{x:.4%}"


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def _annualized_return(total_return: float, n_days: int) -> float:
    if n_days <= 0:
        return float("nan")
    base = 1.0 + total_return
    if base <= 0:
        return float("nan")
    return base ** (TRADING_DAYS_PER_YEAR / n_days) - 1.0


def _sharpe(daily_returns: pd.Series) -> float:
    daily_returns = daily_returns.dropna()
    if len(daily_returns) < 2:
        return float("nan")
    std = daily_returns.std(ddof=1)
    if std == 0 or pd.isna(std):
        return float("nan")
    return float((daily_returns.mean() / std) * math.sqrt(TRADING_DAYS_PER_YEAR))


def _calmar(annualized_return: float, max_drawdown: float) -> float:
    if pd.isna(annualized_return) or pd.isna(max_drawdown) or max_drawdown == 0:
        return float("nan")
    return float(annualized_return / abs(max_drawdown))


def load_input(path: Path, top_n: int, exit_mode: str = DEFAULT_EXIT_MODE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Backtest input not found: {path}")
    df = pd.read_csv(path)
    if "selected_top_n" in df.columns:
        selected = df["selected_top_n"].astype(str).str.lower() == "true"
        df = df.loc[selected].copy()
    if "rank_in_day" in df.columns:
        df = df.loc[pd.to_numeric(df["rank_in_day"], errors="coerce") <= top_n].copy()

    if exit_mode not in EXIT_COLUMN_MAP:
        raise ValueError(f"Unsupported exit_mode={exit_mode!r}; choose from {sorted(EXIT_COLUMN_MAP)}")
    exit_col = EXIT_COLUMN_MAP[exit_mode]

    needed = ["trade_date", "next_trade_date", "ts_code", exit_col, "rank_in_day"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["next_trade_date"] = pd.to_datetime(df["next_trade_date"])
    df[exit_col] = pd.to_numeric(df[exit_col], errors="coerce")
    df = df.dropna(subset=["trade_date", "next_trade_date", exit_col])
    if df.empty:
        raise ValueError("Input has no valid selected rows after filtering")
    df = df.copy()
    df["realized_exit_mode"] = exit_mode
    df["realized_return_column"] = exit_col
    df["realized_return_raw"] = df[exit_col]
    return df.sort_values(["trade_date", "rank_in_day", "ts_code"]).reset_index(drop=True)


def run_backtest(
    df: pd.DataFrame,
    initial_capital: float,
    fee_bps: float,
    slippage_bps: float,
    top_n: int,
    exit_mode: str = DEFAULT_EXIT_MODE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    round_trip_cost = (fee_bps + slippage_bps) / 10000.0
    if exit_mode not in EXIT_COLUMN_MAP:
        raise ValueError(f"Unsupported exit_mode={exit_mode!r}; choose from {sorted(EXIT_COLUMN_MAP)}")

    daily_rows = []
    trade_rows = []
    capital = initial_capital

    for trade_date, day_df in df.groupby("trade_date", sort=True):
        day_df = day_df.sort_values(["rank_in_day", "ts_code"]).head(top_n).copy()
        n = len(day_df)
        if n == 0:
            continue
        alloc_per_name = capital / n

        if "realized_return_raw" in day_df.columns:
            raw_returns = pd.to_numeric(day_df["realized_return_raw"], errors="coerce")
        else:
            exit_col = EXIT_COLUMN_MAP.get(exit_mode)
            if exit_col is None or exit_col not in day_df.columns:
                raise ValueError(
                    f"Input is missing both 'realized_return_raw' and exit column {exit_col!r} for exit_mode={exit_mode!r}"
                )
            raw_returns = pd.to_numeric(day_df[exit_col], errors="coerce")

        realized_returns = raw_returns - round_trip_cost
        realized_returns = realized_returns.fillna(0.0)
        pnl_each = alloc_per_name * realized_returns
        end_values = alloc_per_name * (1.0 + realized_returns)

        day_trade_df = day_df.copy()
        day_trade_df["alloc_capital"] = alloc_per_name
        day_trade_df["round_trip_cost"] = round_trip_cost
        day_trade_df["realized_return_raw"] = raw_returns.values
        day_trade_df["realized_return_net"] = realized_returns.values
        day_trade_df["pnl"] = pnl_each.values
        day_trade_df["end_capital"] = end_values.values
        trade_rows.append(day_trade_df)

        day_start = capital
        day_pnl = float(pnl_each.sum())
        capital = float(day_start + day_pnl)
        day_ret = day_pnl / day_start if day_start else float("nan")

        daily_rows.append({
            "trade_date": trade_date,
            "next_trade_date": day_df["next_trade_date"].iloc[0],
            "n_positions": n,
            "gross_mean_return": float(raw_returns.mean()),
            "net_mean_return": float(realized_returns.mean()),
            "day_pnl": day_pnl,
            "day_return": day_ret,
            "start_capital": day_start,
            "end_capital": capital,
        })

    daily = pd.DataFrame(daily_rows)
    trades = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()

    if daily.empty:
        raise ValueError("Backtest produced no daily rows")

    total_return = daily["end_capital"].iloc[-1] / initial_capital - 1.0
    annualized = _annualized_return(total_return, len(daily))
    max_dd = _max_drawdown(daily["end_capital"])
    sharpe = _sharpe(daily["day_return"])
    calmar = _calmar(annualized, max_dd)
    win_rate = float((trades["realized_return_net"] > 0).mean()) if not trades.empty else float("nan")
    avg_trade = float(trades["realized_return_net"].mean()) if not trades.empty else float("nan")

    summary = {
        "initial_capital": initial_capital,
        "ending_capital": float(daily["end_capital"].iloc[-1]),
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "max_drawdown": float(max_dd),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "trade_days": int(len(daily)),
        "trade_count": int(len(trades)),
        "avg_positions_per_day": float(daily["n_positions"].mean()),
        "win_rate": float(win_rate),
        "avg_trade_return_net": float(avg_trade),
        "fee_bps": float(fee_bps),
        "slippage_bps": float(slippage_bps),
        "top_n": int(top_n),
        "exit_mode": exit_mode,
    }
    return daily, trades, summary


def write_summary(path: Path, input_path: Path, daily: pd.DataFrame, trades: pd.DataFrame, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Top-N Overnight Baseline Backtest Summary

- Input: `{input_path}`
- Trade window: `{daily['trade_date'].min().date()}` -> `{daily['trade_date'].max().date()}`
- Next-open exits through: `{daily['next_trade_date'].max().date()}`
- Trade days: `{summary['trade_days']}`
- Trade count: `{summary['trade_count']}`
- Exit mode: `{summary.get('exit_mode', DEFAULT_EXIT_MODE)}`
- Top-N: `{summary['top_n']}`
- Fee (bps): `{summary['fee_bps']}`
- Slippage (bps): `{summary['slippage_bps']}`

## Performance
- Initial capital: `{summary['initial_capital']:.2f}`
- Ending capital: `{summary['ending_capital']:.2f}`
- Total return: `{_fmt_pct(summary['total_return'])}`
- Annualized return: `{_fmt_pct(summary['annualized_return']) if not pd.isna(summary['annualized_return']) else 'nan'}`
- Max drawdown: `{_fmt_pct(summary['max_drawdown']) if not pd.isna(summary['max_drawdown']) else 'nan'}`
- Sharpe: `{summary['sharpe']:.4f}`
- Calmar: `{summary['calmar']:.4f}`
- Win rate: `{_fmt_pct(summary['win_rate']) if not pd.isna(summary['win_rate']) else 'nan'}`
- Avg trade return (net): `{_fmt_pct(summary['avg_trade_return_net']) if not pd.isna(summary['avg_trade_return_net']) else 'nan'}`
- Avg positions/day: `{summary['avg_positions_per_day']:.2f}`

## Daily return snapshot
{daily.head(10).to_markdown(index=False)}

## Top trade snapshot
{trades[['trade_date', 'ts_code', 'rank_in_day', 'realized_return_raw', 'realized_return_net', 'pnl']].head(15).to_markdown(index=False) if not trades.empty else 'No trades'}
"""
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run first-pass Top-N overnight baseline backtest")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Top-N baseline input CSV")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top-N positions per day")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    parser.add_argument("--exit-mode", choices=sorted(EXIT_COLUMN_MAP.keys()), default=DEFAULT_EXIT_MODE, help="Exit return column to backtest")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = DEFAULT_OUTDIR
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_input(input_path, args.top_n, exit_mode=args.exit_mode)
    daily, trades, summary = run_backtest(
        df=df,
        initial_capital=args.initial_capital,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        top_n=args.top_n,
        exit_mode=args.exit_mode,
    )

    stem = input_path.stem.replace("topn_baseline_input_", "") + f"_top{args.top_n}_{args.exit_mode}"
    daily_path = outdir / f"topn_backtest_daily_{stem}.csv"
    trades_path = outdir / f"topn_backtest_trades_{stem}.csv"
    summary_path = outdir / f"topn_backtest_summary_{stem}.md"

    daily.to_csv(daily_path, index=False)
    trades.to_csv(trades_path, index=False)
    write_summary(summary_path, input_path, daily, trades, summary)

    print(f"Wrote daily results: {daily_path} rows={len(daily)}")
    print(f"Wrote trade results: {trades_path} rows={len(trades)}")
    print(f"Wrote summary: {summary_path}")
    print(f"Total return={summary['total_return']:.6f} Sharpe={summary['sharpe']:.4f} MaxDD={summary['max_drawdown']:.6f}")


if __name__ == "__main__":
    main()
