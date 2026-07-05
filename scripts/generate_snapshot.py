#!/usr/bin/env python3.12
"""Quick snapshot generator for live overnight pipeline.

Uses Tencent qt.gtimg.cn (free, no auth, no WAF).
Loads universe from the feature table, fetches realtime quotes,
and writes a standardized snapshot CSV.

Usage:
    python3.12 scripts/generate_snapshot.py --trade-date 2026-06-30
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.default_config import DEFAULT_CONFIG
from dataflows.realtime_snapshot_provider import (
    load_universe_from_feature_table,
    fetch_tencent_realtime_snapshot,
    write_snapshot,
    assess_snapshot_quality,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trade-date", required=True)
    p.add_argument("--history-feature-table", default=DEFAULT_CONFIG["overnight_feature_table_path"])
    p.add_argument("--out-dir", default="data/snapshots")
    p.add_argument("--chunk-size", type=int, default=80)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trade_date_fmt = args.trade_date.replace("-", "")
    snapshot_path = out_dir / f"snapshot_{trade_date_fmt}_tencent.csv"
    manifest_path = out_dir / f"snapshot_{trade_date_fmt}_tencent.manifest.json"

    print(f"Loading universe from {args.history_feature_table}...")
    universe = load_universe_from_feature_table(args.history_feature_table, trade_date=args.trade_date)
    print(f"Universe: {len(universe)} stocks")

    print(f"Fetching Tencent realtime snapshot ({len(universe)} stocks, chunk_size={args.chunk_size})...")
    snapshot = fetch_tencent_realtime_snapshot(universe, chunk_size=args.chunk_size)
    print(f"Snapshot: {len(snapshot)} rows")

    quality = assess_snapshot_quality(snapshot, universe)
    print(f"Coverage: {quality.usable_count}/{quality.expected_count} ({quality.coverage_ratio:.1%})")
    print(f"Quote time range: {quality.min_quote_time} - {quality.max_quote_time}")

    write_snapshot(snapshot, snapshot_path)
    print(f"Wrote: {snapshot_path}")

    manifest = {
        "trade_date": args.trade_date,
        "source": "tencent",
        "run_ts": datetime.now().isoformat(timespec="seconds"),
        "universe_count": len(universe),
        "snapshot_path": str(snapshot_path),
        "quality": {
            "expected_count": quality.expected_count,
            "returned_count": quality.returned_count,
            "usable_count": quality.usable_count,
            "coverage_ratio": quality.coverage_ratio,
            "min_quote_time": quality.min_quote_time,
            "max_quote_time": quality.max_quote_time,
            "freshness_ok": quality.freshness_ok,
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")

    if not quality.ok:
        print(f"WARNING: snapshot quality degraded (coverage {quality.coverage_ratio:.1%}, freshness_ok={quality.freshness_ok})")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
