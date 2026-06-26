#!/usr/bin/env python3
"""One-command 14:55 live overnight run.

This is the operational wrapper for the live overnight path:

1. Build/load universe.
2. Fetch Tushare realtime snapshot.
3. Validate coverage and quote freshness.
4. Run deterministic live overnight inference.
5. Persist snapshot, Top-N outputs, summary, and manifest in one run directory.

Default behavior is strict for real execution: stale quotes or low coverage abort
before inference.  Use --allow-stale only for smoke tests or non-trading dry runs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from config.default_config import DEFAULT_CONFIG
from dataflows.overnight_live_provider import run_live_inference
from dataflows.realtime_snapshot_provider import (
    assess_snapshot_quality,
    fetch_tushare_realtime_snapshot,
    load_universe_from_feature_table,
    normalize_ts_code,
    write_snapshot,
)


DEFAULT_OUT_ROOT = Path("data/overnight_live_1455")


def _fmt_date(value: str) -> str:
    return str(value).replace("-", "")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_universe(args) -> list[str]:
    if args.symbols:
        return sorted({normalize_ts_code(x) for x in args.symbols.split(",") if x.strip()})
    if args.universe_csv:
        p = Path(args.universe_csv)
        if not p.exists():
            raise FileNotFoundError(f"Universe CSV not found: {p}")
        df = pd.read_csv(p)
        if args.universe_column not in df.columns:
            raise ValueError(f"Universe CSV missing column {args.universe_column!r}: {p}")
        return sorted({normalize_ts_code(x) for x in df[args.universe_column].dropna().astype(str)})
    return load_universe_from_feature_table(args.history_feature_table, trade_date=args.trade_date)


def _select_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in [
            "rank_in_live_day", "ts_code", "name", "industry", "market",
            "overnight_live_score", "final_live_score",
            "heavy_score", "heavy_adjustment", "heavy_tier", "heavy_veto", "heavy_keep_rank",
            "agent_score", "agent_adjustment", "agent_risk_level", "agent_veto", "last_price", "open", "high", "low", "pre_close",
            "live_return_vs_pre_close", "live_return_vs_prev_close", "from_day_high", "live_range_pos",
            "hist_ret_close_1d", "hist_ret_close_3d", "hist_ret_close_5d",
            "hist_overnight_prev_1d", "hist_overnight_prev_3d_mean",
            "hist_overnight_positive_rate_5d", "live_pass_risk_filter", "live_reject_reasons",
        ] if c in df.columns
    ]


def _write_live_outputs(result: dict, run_dir: Path, top_n: int, candidate_pool_size: int) -> dict[str, Path]:
    suffix = f"{_fmt_date(result['trade_date'])}_top{top_n}_pool{candidate_pool_size}"
    paths = {
        "features": run_dir / f"live_features_{suffix}.csv",
        "scored": run_dir / f"live_scored_{suffix}.csv",
        "candidate_pool": run_dir / f"live_candidate_pool_{suffix}.csv",
        "selected": run_dir / f"live_selected_{suffix}.csv",
        "summary": run_dir / f"live_summary_{suffix}.md",
    }
    result["features"].to_csv(paths["features"], index=False)
    result["scored"].to_csv(paths["scored"], index=False)
    result["candidate_pool"].to_csv(paths["candidate_pool"], index=False)
    result["selected"].to_csv(paths["selected"], index=False)

    selected_cols = _select_columns(result["selected"])
    pool_cols = _select_columns(result["candidate_pool"])
    lines = [
        "# 14:55 Live Overnight Run Summary",
        "",
        f"- trade_date: `{result['trade_date']}`",
        f"- top_n: `{top_n}`",
        f"- candidate_pool_size: `{candidate_pool_size}`",
        "- exit_rule: `next_open_sell`",
        "- future_label_usage: `none`",
        f"- snapshot_csv: `{result['snapshot_csv']}`",
        f"- history_feature_table_path: `{result['history_feature_table_path']}`",
        "",
        "## Selected",
        "",
    ]
    if result["selected"].empty:
        lines.append("- No candidates passed live risk filters.")
    else:
        lines.append(result["selected"][selected_cols].to_markdown(index=False))
    lines.extend(["", "## Candidate Pool", ""])
    if result["candidate_pool"].empty:
        lines.append("- Candidate pool is empty.")
    else:
        lines.append(result["candidate_pool"][pool_cols].to_markdown(index=False))
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one-command 14:55 live overnight Top-N workflow")
    parser.add_argument("--trade-date", required=True, help="Decision date, YYYY-MM-DD")
    parser.add_argument("--history-feature-table", default=DEFAULT_CONFIG["overnight_feature_table_path"], help="Historical feature table CSV")
    parser.add_argument("--symbols", default=None, help="Optional comma-separated ts_codes/bare symbols for smoke tests")
    parser.add_argument("--universe-csv", default=None, help="Optional universe CSV")
    parser.add_argument("--universe-column", default="ts_code", help="Universe CSV symbol column")
    parser.add_argument("--top-n", type=int, default=5, help="Number of final picks")
    parser.add_argument("--candidate-pool-size", type=int, default=20, help="Candidate pool size")
    parser.add_argument("--agent-review-scores", default=None, help="Optional legacy/light TradingAgents2.0 pre-close review scores CSV for final fusion")
    parser.add_argument("--heavy-review-scores", default=None, help="Optional heavy Top50->Top15 review scores CSV")
    parser.add_argument("--light-review-scores", default=None, help="Optional light/fast Top15 review scores CSV")
    parser.add_argument("--live-weight", type=float, default=0.75, help="Final fusion weight for deterministic live score")
    parser.add_argument("--agent-weight", type=float, default=0.25, help="Legacy light-only fusion weight for agent_score")
    parser.add_argument("--heavy-weight", type=float, default=0.25, help="Multi-stage fusion weight for heavy_score")
    parser.add_argument("--light-weight", type=float, default=0.15, help="Multi-stage fusion weight for light agent_score")
    parser.add_argument("--chunk-size", type=int, default=300, help="Tushare realtime request chunk size")
    parser.add_argument("--min-coverage", type=float, default=0.95, help="Minimum usable snapshot coverage")
    parser.add_argument("--min-quote-time", default="14:54:00", help="Minimum acceptable max quote_time for formal execution")
    parser.add_argument("--allow-stale", action="store_true", help="Continue despite stale quote_time; for smoke tests only")
    parser.add_argument("--allow-low-coverage", action="store_true", help="Continue despite low coverage; for smoke tests only")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root")
    args = parser.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / args.trade_date / run_ts
    _ensure_dir(run_dir)

    universe = _load_universe(args)
    snapshot = fetch_tushare_realtime_snapshot(universe, chunk_size=args.chunk_size)
    quality = assess_snapshot_quality(snapshot, universe, stale_time_threshold=args.min_quote_time)

    snapshot_path = run_dir / f"snapshot_{_fmt_date(args.trade_date)}_{run_ts}_tushare.csv"
    write_snapshot(snapshot, snapshot_path)

    abort_reasons = []
    if quality.coverage_ratio < args.min_coverage and not args.allow_low_coverage:
        abort_reasons.append(f"coverage {quality.coverage_ratio:.2%} < min_coverage {args.min_coverage:.2%}")
    if not quality.freshness_ok and not args.allow_stale:
        abort_reasons.append(f"stale quote_time max={quality.max_quote_time} < min_quote_time {args.min_quote_time}")

    manifest = {
        "trade_date": args.trade_date,
        "run_ts": run_ts,
        "source": "tushare.get_realtime_quotes",
        "universe_count": len(universe),
        "snapshot_path": str(snapshot_path),
        "history_feature_table": args.history_feature_table,
        "top_n": args.top_n,
        "candidate_pool_size": args.candidate_pool_size,
        "agent_review_scores": args.agent_review_scores,
        "heavy_review_scores": args.heavy_review_scores,
        "light_review_scores": args.light_review_scores,
        "live_weight": args.live_weight,
        "agent_weight": args.agent_weight,
        "heavy_weight": args.heavy_weight,
        "light_weight": args.light_weight,
        "min_coverage": args.min_coverage,
        "min_quote_time": args.min_quote_time,
        "allow_stale": args.allow_stale,
        "allow_low_coverage": args.allow_low_coverage,
        "quality": {
            "expected_count": quality.expected_count,
            "returned_count": quality.returned_count,
            "usable_count": quality.usable_count,
            "coverage_ratio": quality.coverage_ratio,
            "min_quote_time": quality.min_quote_time,
            "max_quote_time": quality.max_quote_time,
            "freshness_ok": quality.freshness_ok,
        },
        "abort_reasons": abort_reasons,
        "outputs": {},
    }

    if abort_reasons:
        manifest["status"] = "aborted_before_inference"
        manifest_path = run_dir / f"manifest_{_fmt_date(args.trade_date)}_{run_ts}.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote snapshot: {snapshot_path} rows={len(snapshot)}")
        print(f"Wrote manifest: {manifest_path}")
        print(
            f"Coverage: {quality.usable_count}/{quality.expected_count} "
            f"({quality.coverage_ratio:.2%}), quote_time={quality.min_quote_time}..{quality.max_quote_time}, "
            f"freshness_ok={quality.freshness_ok}"
        )
        raise SystemExit("; ".join(abort_reasons))

    result = run_live_inference(
        snapshot_csv=snapshot_path,
        trade_date=args.trade_date,
        history_feature_table_path=args.history_feature_table,
        top_n=args.top_n,
        candidate_pool_size=args.candidate_pool_size,
        review_scores_path=args.agent_review_scores,
        heavy_review_scores_path=args.heavy_review_scores,
        light_review_scores_path=args.light_review_scores,
        live_weight=args.live_weight,
        agent_weight=args.agent_weight,
        heavy_weight=args.heavy_weight,
        light_weight=args.light_weight,
    )
    output_paths = _write_live_outputs(result, run_dir, args.top_n, args.candidate_pool_size)
    manifest["status"] = "completed"
    manifest["rows"] = {
        "snapshot": int(len(snapshot)),
        "features": int(len(result["features"])),
        "scored": int(len(result["scored"])),
        "candidate_pool": int(len(result["candidate_pool"])),
        "selected": int(len(result["selected"])),
    }
    manifest["outputs"] = {k: str(v) for k, v in output_paths.items()}
    manifest_path = run_dir / f"manifest_{_fmt_date(args.trade_date)}_{run_ts}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote snapshot: {snapshot_path} rows={len(snapshot)}")
    print(f"Coverage: {quality.usable_count}/{quality.expected_count} ({quality.coverage_ratio:.2%}), quote_time={quality.min_quote_time}..{quality.max_quote_time}, freshness_ok={quality.freshness_ok}")
    for key, path in output_paths.items():
        print(f"Wrote {key}: {path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
