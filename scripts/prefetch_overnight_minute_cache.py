#!/usr/bin/env python3
"""Prefetch/cache minute windows for overnight candidate pools.

This script is intentionally separate from decision-time ranking so minute API
pressure can be spread across the afternoon and all downstream steps can read
from local cache.

Typical use:
1. Build or supply a candidate universe CSV (Top50 / Top100 / Top300).
2. Run this script every 10-15 minutes between 14:00 and 15:00.
3. Decision-time pipeline reads local minute cache first.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

from dataflows.overnight_minute_cache import (
    DEFAULT_MINUTE_CACHE_DIR,
    fetch_minute_window_frame,
    load_cached_minute_window_frame,
    minute_cache_path,
    normalize_date,
)
from dataflows.realtime_snapshot_provider import normalize_ts_code


DEFAULT_OUT_ROOT = Path("data/overnight_mvp/minute_prefetch_runs")
DEFAULT_CANDIDATE_LIMIT = 100


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_universe(args) -> pd.DataFrame:
    if args.candidate_csv:
        p = Path(args.candidate_csv)
        if not p.exists():
            raise FileNotFoundError(f"Candidate CSV not found: {p}")
        df = pd.read_csv(p)
    elif args.snapshot_csv:
        p = Path(args.snapshot_csv)
        if not p.exists():
            raise FileNotFoundError(f"Snapshot CSV not found: {p}")
        df = pd.read_csv(p)
    else:
        raise ValueError("Either --candidate-csv or --snapshot-csv is required")

    symbol_col = args.symbol_column
    if symbol_col not in df.columns:
        raise ValueError(f"Input file missing symbol column {symbol_col!r}")
    out = df.copy()
    out["ts_code"] = out[symbol_col].astype(str).map(normalize_ts_code)
    if args.rank_column and args.rank_column in out.columns:
        out = out.sort_values(args.rank_column, ascending=True)
    elif args.score_column and args.score_column in out.columns:
        out = out.sort_values(args.score_column, ascending=False)
    out = out.dropna(subset=["ts_code"]).drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
    return out


def _apply_pool_controls(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    if args.candidate_limit and args.candidate_limit > 0:
        out = out.head(int(args.candidate_limit)).copy()
    if args.shard_count and args.shard_count > 1:
        shard_id = int(args.shard_id)
        shard_count = int(args.shard_count)
        rows = []
        for row in out.itertuples(index=False):
            bucket = sum(ord(ch) for ch in str(row.ts_code)) % shard_count
            if bucket == shard_id:
                rows.append(row._asdict())
        out = pd.DataFrame(rows) if rows else pd.DataFrame(columns=out.columns)
    return out.reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Prefetch/cache minute windows for overnight candidate pools")
    p.add_argument("--trade-date", required=True, help="Decision date YYYY-MM-DD")
    p.add_argument("--candidate-csv", default=None, help="CSV of candidate pool symbols/ranks/scores")
    p.add_argument("--snapshot-csv", default=None, help="Optional snapshot CSV used as raw universe input")
    p.add_argument("--symbol-column", default="ts_code", help="Symbol column in input CSV")
    p.add_argument("--rank-column", default="rank_in_live_day", help="Ascending rank column for prioritization")
    p.add_argument("--score-column", default="overnight_live_score", help="Descending score column when rank column absent")
    p.add_argument("--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT, help="Max number of symbols to prefetch")
    p.add_argument("--start-time", default="14:30:00")
    p.add_argument("--end-time", default="15:00:00")
    p.add_argument("--freq", default="5min")
    p.add_argument("--cache-dir", default=str(DEFAULT_MINUTE_CACHE_DIR), help="Minute cache directory")
    p.add_argument("--missing-only", action="store_true", help="Skip remote fetch when cache file already exists")
    p.add_argument("--force-refresh", action="store_true", help="Force refresh even if cache exists")
    p.add_argument("--max-symbols", type=int, default=0, help="Optional extra hard cap after sharding")
    p.add_argument("--shard-id", type=int, default=0, help="Shard id in [0, shard-count)")
    p.add_argument("--shard-count", type=int, default=1, help="Number of shards for time spreading")
    p.add_argument("--source", default="eastmoney,tushare", help="Comma-separated minute sources. Live default prefers Eastmoney, then Tushare only as fallback/offline supplement.")
    p.add_argument("--stop-on-rate-limit", action="store_true", default=True, help="Fast-fuse the whole prefetch after the first Tushare rate-limit error")
    p.add_argument("--no-stop-on-rate-limit", dest="stop_on_rate_limit", action="store_false")
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = p.parse_args()

    trade_date = normalize_date(args.trade_date)
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = Path.cwd() / cache_dir
    out_root = Path(args.out_root)
    if not out_root.is_absolute():
        out_root = Path.cwd() / out_root
    run_dir = out_root / trade_date / datetime.now().strftime("%Y%m%d_%H%M%S")
    _ensure_dir(run_dir)

    universe = _apply_pool_controls(_load_universe(args), args)
    if args.max_symbols and args.max_symbols > 0:
        universe = universe.head(int(args.max_symbols)).copy()

    results: list[dict[str, object]] = []
    stats = {"ok": 0, "cached": 0, "empty": 0, "error": 0, "cache_miss": 0, "rate_limited": 0, "skipped_after_rate_limit": 0}
    stopped_early = False
    stop_reason = ""

    for idx, row in enumerate(universe.itertuples(index=False), start=1):
        ts_code = str(getattr(row, "ts_code"))
        if stopped_early:
            stats["skipped_after_rate_limit"] += 1
            results.append({
                "seq": idx,
                "ts_code": ts_code,
                "trade_date": trade_date,
                "status": "skipped_after_rate_limit",
                "row_count": 0,
                "cache_path": "",
                "error": stop_reason,
            })
            continue
        cache_path = minute_cache_path(cache_dir, ts_code, trade_date, args.freq, args.start_time, args.end_time)
        already_exists = cache_path.exists()

        if args.missing_only and already_exists and not args.force_refresh:
            df, err = load_cached_minute_window_frame(ts_code, trade_date, cache_dir, start_time=args.start_time, end_time=args.end_time, freq=args.freq)
            status = "cached" if err is None else "cache_read_error"
            stats["cached" if err is None else "error"] += 1
            results.append({
                "seq": idx,
                "ts_code": ts_code,
                "trade_date": trade_date,
                "status": status,
                "row_count": 0 if df is None else int(len(df)),
                "cache_path": str(cache_path),
                "error": err or "",
            })
            continue

        df, err = fetch_minute_window_frame(
            ts_code=ts_code,
            trade_date=trade_date,
            cache_dir=cache_dir,
            start_time=args.start_time,
            end_time=args.end_time,
            freq=args.freq,
            force_refresh=args.force_refresh,
            allow_remote=True,
            source=args.source,
            max_retries=0,
        )
        if err is None:
            status = "ok" if not already_exists or args.force_refresh else "cached"
            stats["ok" if status == "ok" else "cached"] += 1
        elif err == "empty":
            status = "empty"
            stats["empty"] += 1
        elif err == "cache_miss":
            status = "cache_miss"
            stats["cache_miss"] += 1
        elif "TushareRateLimitError" in str(err) or "频率超限" in str(err) or "1次/小时" in str(err):
            status = "rate_limited"
            stats["rate_limited"] += 1
            if args.stop_on_rate_limit:
                stopped_early = True
                stop_reason = str(err)
        else:
            status = "error"
            stats["error"] += 1

        results.append({
            "seq": idx,
            "ts_code": ts_code,
            "trade_date": trade_date,
            "status": status,
            "row_count": 0 if df is None else int(len(df)),
            "cache_path": str(cache_path),
            "error": err or "",
        })

    result_df = pd.DataFrame(results)
    result_csv = run_dir / f"minute_prefetch_result_{trade_date.replace('-', '')}.csv"
    result_df.to_csv(result_csv, index=False)

    manifest = {
        "trade_date": trade_date,
        "candidate_count": int(len(universe)),
        "cache_dir": str(cache_dir),
        "candidate_csv": args.candidate_csv,
        "snapshot_csv": args.snapshot_csv,
        "symbol_column": args.symbol_column,
        "rank_column": args.rank_column,
        "score_column": args.score_column,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "freq": args.freq,
        "candidate_limit": args.candidate_limit,
        "max_symbols": args.max_symbols,
        "shard_id": args.shard_id,
        "shard_count": args.shard_count,
        "missing_only": args.missing_only,
        "force_refresh": args.force_refresh,
        "source": args.source,
        "stop_on_rate_limit": bool(args.stop_on_rate_limit),
        "stopped_early": bool(stopped_early),
        "stop_reason": stop_reason,
        "degraded": bool(stopped_early or stats.get("rate_limited", 0) > 0 or stats.get("error", 0) > 0),
        "result_csv": str(result_csv),
    }
    manifest_path = run_dir / "minute_prefetch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote minute prefetch result: {result_csv}")
    print(f"Wrote minute prefetch manifest: {manifest_path}")
    print(f"Stats: {stats}")


if __name__ == "__main__":
    main()
