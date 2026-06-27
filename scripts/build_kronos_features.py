#!/usr/bin/env python3
"""Offline batch runner: pre-compute Kronos features for overnight pipeline.

Usage:
  # Must use the venv with Kronos deps installed
  .venv-kronos/bin/python scripts/build_kronos_features.py \
    --trade-date 2026-06-27 \
    --symbols 000001,000002,000333,600519,600036 \
    --out data/kronos/kronos_features_20260627.csv

  # Or feed symbols from a candidate pool CSV:
  .venv-kronos/bin/python scripts/build_kronos_features.py \
    --trade-date 2026-06-27 \
    --candidate-pool data/overnight_live_multistage/2026-06-27/.../live_candidate_pool_*.csv \
    --out data/kronos/kronos_features_20260627.csv

Run this BEFORE market open (e.g. 08:00-09:00).  It downloads historical data
via AkShare and runs Kronos GPU inference.  The output CSV is read by the live
14:30-14:57 pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure parent is on path so we can import dataflows
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pandas as pd

from dataflows.kronos_provider import (
    build_kronos_batch_features,
    write_kronos_features,
)

DEFAULT_OUT_DIR = Path("data/kronos")
DEFAULT_MODEL = "NeoQuasar/Kronos-small"
DEFAULT_LOOKBACK = 200
DEFAULT_PRED_LEN = 5
MAX_RETRIES_PER_SYMBOL = 3


def _load_symbols_from_csv(path: str | Path, col: str = "ts_code") -> list[str]:
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"CSV missing column {col!r}: {path}")
    codes = df[col].dropna().astype(str).unique().tolist()
    # Normalize: strip exchange suffix if needed
    return sorted(codes)


def main():
    ap = argparse.ArgumentParser(description="Pre-compute Kronos features offline")
    ap.add_argument("--trade-date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--symbols", default=None, help="Comma-separated ts_codes")
    ap.add_argument("--candidate-pool", default=None, help="CSV with ts_code column")
    ap.add_argument("--symbol-column", default="ts_code")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Kronos model name on HuggingFace")
    ap.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    ap.add_argument("--pred-len", type=int, default=DEFAULT_PRED_LEN)
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--manifest", default=None, help="Output manifest JSON path")
    ap.add_argument("--max-symbols", type=int, default=0, help="Cap symbol count (0=unlimited)")
    args = ap.parse_args()

    # Resolve symbols
    if args.symbols:
        codes = [x.strip() for x in args.symbols.split(",") if x.strip()]
    elif args.candidate_pool:
        codes = _load_symbols_from_csv(args.candidate_pool, args.symbol_column)
    else:
        raise SystemExit("Must specify --symbols or --candidate-pool")

    if args.max_symbols > 0:
        codes = codes[:args.max_symbols]

    if not codes:
        raise SystemExit("No symbols to process")

    print(f"Kronos batch: {len(codes)} symbols, model={args.model}")
    print(f"Symbols: {codes[:10]}{'...' if len(codes) > 10 else ''}")

    # Run batch
    started = time.time()
    result = build_kronos_batch_features(
        ts_codes=codes,
        trade_date=args.trade_date,
        model_name=args.model,
        lookback=args.lookback,
        pred_len=args.pred_len,
    )

    # Write CSV
    out = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"kronos_features_{args.trade_date.replace('-', '')}.csv"
    write_kronos_features(result, out)
    print(f"Wrote features: {out} rows={len(result.features)}")

    # Write manifest
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote manifest: {manifest_path}")

    # Print summary
    s = result.summary
    print(
        f"Done: {s.get('success_rate', 0):.1%} success "
        f"({len(codes) - result.degraded_count}/{len(codes)}), "
        f"degraded={result.degraded_count}, "
        f"elapsed={result.elapsed_seconds:.0f}s"
    )


if __name__ == "__main__":
    main()
