#!/usr/bin/env python3
"""Fetch a batch realtime quote snapshot for live overnight inference.

Default universe source is the latest symbol set in the historical overnight
feature table before --trade-date.  The output CSV is directly consumable by
`scripts/run_overnight_live_inference.py`.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.default_config import DEFAULT_CONFIG
from dataflows.realtime_snapshot_provider import (
    assess_snapshot_quality,
    fetch_realtime_snapshot_with_fallback,
    fetch_tushare_realtime_snapshot,
    fetch_tencent_realtime_snapshot,
    load_universe_from_feature_table,
    normalize_ts_code,
    write_snapshot,
)


DEFAULT_OUT_DIR = Path("data/live_snapshots")


def _fmt_date(value: str) -> str:
    return str(value).replace("-", "")


def _load_universe(args) -> list[str]:
    if args.symbols:
        return sorted({normalize_ts_code(x) for x in args.symbols.split(",") if x.strip()})
    if args.universe_csv:
        p = Path(args.universe_csv)
        if not p.exists():
            raise FileNotFoundError(f"Universe CSV not found: {p}")
        df = pd.read_csv(p)
        col = args.universe_column
        if col not in df.columns:
            raise ValueError(f"Universe CSV missing column {col!r}: {p}")
        return sorted({normalize_ts_code(x) for x in df[col].dropna().astype(str)})
    return load_universe_from_feature_table(args.history_feature_table, trade_date=args.trade_date)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch realtime snapshot for 14:55 live overnight inference")
    parser.add_argument("--trade-date", required=True, help="Decision date, YYYY-MM-DD")
    parser.add_argument("--source", default="tencent", choices=["tencent", "tushare", "auto"], help="Realtime source; auto uses Tencent primary + Tushare fallback")
    parser.add_argument("--fallback-source", default="tushare", choices=["tushare", "tencent", "none"], help="Fallback source when --source=tencent/auto misses symbols")
    parser.add_argument("--symbols", default=None, help="Comma-separated ts_codes or bare symbols")
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV")
    parser.add_argument("--universe-column", default="ts_code", help="Universe CSV column name")
    parser.add_argument("--history-feature-table", default=DEFAULT_CONFIG["overnight_feature_table_path"], help="Historical feature table for default universe")
    parser.add_argument("--chunk-size", type=int, default=300, help="Realtime quote request chunk size")
    parser.add_argument("--min-coverage", type=float, default=0.95, help="Warn/fail below this usable coverage ratio")
    parser.add_argument("--min-quote-time", default=None, help="Warn/fail if max quote_time is older than this HH:MM:SS, e.g. 14:54:00")
    parser.add_argument("--fail-under-coverage", action="store_true", help="Exit non-zero if coverage is below --min-coverage")
    parser.add_argument("--fail-stale", action="store_true", help="Exit non-zero if quote_time is older than --min-quote-time")
    parser.add_argument("--out", default=None, help="Output snapshot CSV path")
    parser.add_argument("--manifest", default=None, help="Output quality manifest JSON path")
    args = parser.parse_args()

    universe = _load_universe(args)
    if not universe:
        raise ValueError("Universe is empty")

    if args.source == "tushare":
        snapshot = fetch_tushare_realtime_snapshot(universe, chunk_size=args.chunk_size)
    elif args.source == "tencent":
        fallback = None if args.fallback_source == "none" else args.fallback_source
        snapshot = fetch_realtime_snapshot_with_fallback(universe, primary="tencent", fallback=fallback, chunk_size=args.chunk_size)
    elif args.source == "auto":
        snapshot = fetch_realtime_snapshot_with_fallback(universe, primary="tencent", fallback="tushare", chunk_size=args.chunk_size)
    else:
        raise ValueError(f"Unsupported source: {args.source}")

    quality = assess_snapshot_quality(snapshot, universe, stale_time_threshold=args.min_quote_time)
    out = Path(args.out) if args.out else DEFAULT_OUT_DIR / f"snapshot_{_fmt_date(args.trade_date)}_{datetime.now().strftime('%H%M%S')}_{args.source}.csv"
    write_snapshot(snapshot, out)

    manifest = {
        "trade_date": args.trade_date,
        "source": args.source,
        "fallback_source": args.fallback_source,
        "source_errors": list(snapshot.attrs.get("errors", [])) if hasattr(snapshot, "attrs") else [],
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "universe_count": len(universe),
        "snapshot_path": str(out),
        "quality": {
            "expected_count": quality.expected_count,
            "returned_count": quality.returned_count,
            "usable_count": quality.usable_count,
            "coverage_ratio": quality.coverage_ratio,
            "min_quote_time": quality.min_quote_time,
            "max_quote_time": quality.max_quote_time,
            "stale_time_threshold": quality.stale_time_threshold,
            "freshness_ok": quality.freshness_ok,
            "ok": quality.coverage_ratio >= args.min_coverage and quality.usable_count > 0 and quality.freshness_ok,
            "min_coverage": args.min_coverage,
        },
    }
    manifest_path = Path(args.manifest) if args.manifest else out.with_suffix(".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote snapshot: {out} rows={len(snapshot)}")
    print(f"Wrote manifest: {manifest_path}")
    print(
        f"Coverage: {quality.usable_count}/{quality.expected_count} "
        f"({quality.coverage_ratio:.2%}), quote_time={quality.min_quote_time}..{quality.max_quote_time}, "
        f"freshness_ok={quality.freshness_ok}"
    )
    if quality.coverage_ratio < args.min_coverage:
        msg = f"coverage below threshold: {quality.coverage_ratio:.2%} < {args.min_coverage:.2%}"
        if args.fail_under_coverage:
            raise SystemExit(msg)
        print(f"WARNING: {msg}")
    if not quality.freshness_ok:
        msg = f"stale quote_time: max_quote_time={quality.max_quote_time} < min_quote_time={args.min_quote_time}"
        if args.fail_stale:
            raise SystemExit(msg)
        print(f"WARNING: {msg}")


if __name__ == "__main__":
    main()
