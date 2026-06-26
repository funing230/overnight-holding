#!/usr/bin/env python3
"""Build a tiny A-share overnight-label demo CSV.

The overnight-holding research labels include:
    T close -> T+1 open / 09:35 / 09:45 / 10:00

Usage examples:
    python scripts/build_overnight_labels_demo.py
    python scripts/build_overnight_labels_demo.py --symbols 000001.SZ,600519.SH --trade-dates 2026-03-24,2026-03-25
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dataflows.ashare_overnight_labels import DEFAULT_EXIT_TIMES, build_overnight_labels


DEFAULT_SYMBOLS = ["000001.SZ", "600519.SH", "300750.SZ", "601318.SH", "000858.SZ"]
DEFAULT_TRADE_DATES = ["2026-03-24", "2026-03-25", "2026-03-26"]
DEFAULT_OUTPUT = Path("data/overnight_labels/overnight_labels_demo.csv")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build demo multi-exit A-share overnight labels.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS), help="Comma-separated A-share symbols")
    parser.add_argument("--trade-dates", default=",".join(DEFAULT_TRADE_DATES), help="Comma-separated T dates, YYYY-MM-DD")
    parser.add_argument("--exit-times", default=",".join(DEFAULT_EXIT_TIMES), help="Comma-separated next-day exit times, e.g. 09:35,09:45,10:00")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    symbols = _split_csv(args.symbols)
    trade_dates = _split_csv(args.trade_dates)
    exit_times = _split_csv(args.exit_times)
    output = Path(args.output)

    df = build_overnight_labels(symbols, trade_dates, exit_times=exit_times)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    print(f"Wrote {len(df)} rows to {output}")
    if "error" in df.columns:
        failed = int(df["error"].notna().sum())
        print(f"Failures: {failed}")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
