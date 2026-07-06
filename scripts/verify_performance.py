#!/usr/bin/env python3
"""Verify yesterday's Top5 predictions against actual market data.

Usage:
  python3 scripts/verify_performance.py 20260630
  python3 scripts/verify_performance.py 20260630 --performance-dir data/performance --show-context
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataflows.performance_tracker import (
    verify_predictions,
    build_performance_context,
    load_history,
)

parser = argparse.ArgumentParser(description="Verify overnight predictions")
parser.add_argument("trade_date", help="Prediction date to verify (YYYYMMDD)")
parser.add_argument("--performance-dir", default="data/performance")
parser.add_argument("--show-context", action="store_true",
                    help="Also print the Selector prompt context block")
args = parser.parse_args()

result = verify_predictions(args.trade_date, data_dir=args.performance_dir)
if result is None:
    print("No data available for verification.")
    sys.exit(1)

print(result.to_string())
print()

if args.show_context:
    # For context, pretend "today" is after the verified date
    from datetime import datetime, timedelta
    next_day = datetime.strptime(args.trade_date, "%Y%m%d") + timedelta(days=1)
    today = next_day.strftime("%Y%m%d")
    print("=== Selector Prompt Context ===")
    print(build_performance_context(today, data_dir=args.performance_dir))

# Summary
total = len(result)
correct = int(result["direction_correct"].sum())
avg_ret = result["total_return"].mean()
tp = int(result["hit_take_profit"].sum())
sl = int(result["hit_stop_loss"].sum())
print(f"\nSummary: {correct}/{total} correct ({correct/total*100:.0f}%), "
      f"avg return {avg_ret:+.2%}, +3%TP:{tp} -2%SL:{sl}")
