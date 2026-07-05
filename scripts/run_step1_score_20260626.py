#!/usr/bin/env python3
"""Build simulated snapshot + history using already-fetched feature data.

Key: compute pre_close (Thursday) from Friday's close and ret_close_1d:
    pre_close = close / (1 + ret_close_1d)

This gives proper live_return_vs_prev_close for the scoring engine.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataflows.overnight_live_provider import (
    load_snapshot_csv,
    build_live_feature_frame,
    score_live_candidates,
    apply_live_risk_filters,
    LiveOvernightConfig,
)

OLD_FEATURE = Path("data/overnight_mvp/features/overnight_features_20260101_20260430.csv")
EXT_FEATURE = Path("data/overnight_mvp/features/overnight_features_ext_20260626.csv")
OUT_DIR = Path("data/output/20260626_weekend")
TRADE_DATE = "2026-06-27"  # Monday (next trading day - the day we're simulating for)

UA = "Mozilla/5.0"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load extended features (Friday June 26 data)
    print("Loading extended features...", flush=True)
    ext = pd.read_csv(EXT_FEATURE)
    print(f"  {len(ext)} stocks", flush=True)

    # Compute pre_close (Thursday) from Friday close and ret_close_1d
    ret = pd.to_numeric(ext["ret_close_1d"], errors="coerce")
    close = pd.to_numeric(ext["close"], errors="coerce")
    ext["pre_close"] = close / (1.0 + ret)
    # Fill NaN pre_close with close (for stocks where ret_close_1d is missing)
    ext["pre_close"] = ext["pre_close"].fillna(close)

    print(f"  pre_close range: {ext['pre_close'].min():.2f} - {ext['pre_close'].max():.2f}")

    # Build snapshot: use Friday's close as last_price
    snap_rows = []
    for _, r in ext.iterrows():
        snap_rows.append({
            "ts_code": r["ts_code"],
            "last_price": r["close"],
            "open": r.get("open", r["close"]),
            "high": r.get("high", r["close"]),
            "low": r.get("low", r["close"]),
            "pre_close": r["pre_close"],
        })

    df_snap = pd.DataFrame(snap_rows)
    snapshot_path = Path("data/snapshots/live_snapshot_20260626.csv")
    df_snap.to_csv(snapshot_path, index=False)
    print(f"  Wrote snapshot: {snapshot_path}", flush=True)

    # Load old feature table for metadata
    print("Loading metadata...", flush=True)
    old = pd.read_csv(OLD_FEATURE)
    old_latest = old.sort_values(["ts_code", "trade_date"]).groupby("ts_code").last().reset_index()
    meta_cols = ["ts_code", "name", "industry", "market", "gap_days",
                 "is_new_listing_180d", "prev_limit_move_like_1d", "prev_soft_outlier_1d"]
    meta_cols = [c for c in meta_cols if c in old_latest.columns]
    meta = old_latest[meta_cols].copy()

    # Merge metadata into features
    merged = ext.merge(meta, on="ts_code", how="left", suffixes=("", "_meta"))
    for col in meta_cols:
        if col == "ts_code":
            continue
        meta_src = f"{col}_meta"
        if meta_src in merged.columns:
            if col in merged.columns:
                merged[col] = merged[col].combine_first(merged[meta_src])
            else:
                merged[col] = merged[meta_src]
            merged = merged.drop(columns=[meta_src])
    merged["name"] = merged["name"].fillna(merged["ts_code"])
    merged["industry"] = merged["industry"].fillna("")
    merged["market"] = merged["market"].fillna("主板")
    merged["gap_days"] = merged["gap_days"].fillna(0).astype(int)
    merged["is_new_listing_180d"] = merged.get("is_new_listing_180d", pd.Series(False, index=merged.index)).fillna(False)
    merged["prev_limit_move_like_1d"] = merged.get("prev_limit_move_like_1d", pd.Series(False, index=merged.index)).fillna(False)
    merged["prev_soft_outlier_1d"] = merged.get("prev_soft_outlier_1d", pd.Series(False, index=merged.index)).fillna(False)

    # Build history_latest — use Thursday's close as history_close
    # by adjusting the history row to represent the PREVIOUS day
    history_latest = merged.rename(
        columns={
            "trade_date": "history_trade_date",
            "pre_close": "history_close",  # Thursday's close = "history close"
            "close": "_friday_close",
            "ret_close_1d": "hist_ret_close_1d",
            "ret_close_3d": "hist_ret_close_3d",
            "ret_close_5d": "hist_ret_close_5d",
            "close_range_pos_5d": "hist_close_range_pos_5d",
            "close_drawdown_10d": "hist_close_drawdown_10d",
            "close_vol_5d": "hist_close_vol_5d",
            "overnight_prev_1d": "hist_overnight_prev_1d",
            "overnight_prev_3d_mean": "hist_overnight_prev_3d_mean",
            "overnight_prev_5d_mean": "hist_overnight_prev_5d_mean",
            "overnight_prev_5d_std": "hist_overnight_prev_5d_std",
            "overnight_positive_rate_5d": "hist_overnight_positive_rate_5d",
        }
    )

    # Run live inference pipeline
    print("Building live feature frame...", flush=True)
    snapshot = load_snapshot_csv(snapshot_path)
    features = build_live_feature_frame(snapshot, history_latest, TRADE_DATE)
    print(f"  {len(features)} stocks", flush=True)

    # Verify live_return is meaningful
    lr = pd.to_numeric(features["live_return_vs_prev_close"], errors="coerce")
    print(f"  live_return_vs_prev_close: min={lr.min():.4f} max={lr.max():.4f} mean={lr.mean():.4f}")

    lr_pre = pd.to_numeric(features.get("live_return_vs_pre_close", lr), errors="coerce")
    print(f"  live_return_vs_pre_close: min={lr_pre.min():.4f} max={lr_pre.max():.4f} mean={lr_pre.mean():.4f}")

    # Score
    print("Scoring...", flush=True)
    scored = score_live_candidates(features)
    cfg = LiveOvernightConfig(history_feature_table_path=EXT_FEATURE)
    scored = apply_live_risk_filters(scored, cfg)

    # Top 50
    top50 = scored.sort_values("overnight_live_score", ascending=False).head(50)
    top50 = top50.reset_index(drop=True)

    # Print
    print("\n" + "="*105)
    print("TOP 20 确定性打分 — 模拟 周一(6/29)买入，基于 周五(6/26) 收盘数据")
    print("="*105)
    print(f"{'#':<3} {'代码':<12} {'名称':<8} {'行业':<12} {'score':>7} {'价格':>8} {'日收益':>8} {'区间%':>7} {'距高%':>7} 风险")
    print("-"*105)
    for i, (_, r) in enumerate(top50.head(20).iterrows(), 1):
        score = r["overnight_live_score"]
        ret = r.get("live_return_vs_prev_close", 0)
        rp = r.get("live_range_pos", 0)
        fh = r.get("from_day_high", 0) if pd.notna(r.get("from_day_high")) else 0
        px = r["last_price"]
        print(f"  {i:<2} {r['ts_code']:<12} {str(r.get('name','')):<8} {str(r.get('industry','')):<12} "
              f"{score:>7.4f} {px:>8.2f} {ret:>8.2%} {rp:>7.1%} {fh:>7.1%}  "
              f"{'✓' if r['live_pass_risk_filter'] else '✗'}")
    print("-"*105)

    # Save
    top50.to_csv(OUT_DIR / "live_candidate_pool_top50.csv", index=False)
    scored.to_csv(OUT_DIR / "live_scored_full.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/live_candidate_pool_top50.csv")

    # Stats
    print(f"Score range: {top50['overnight_live_score'].min():.4f} - {top50['overnight_live_score'].max():.4f}")
    print(f"Pass risk: {top50['live_pass_risk_filter'].sum()}/{len(top50)}")

    # Full Top50
    print(f"\nFull Top50:")
    for _, r in top50.iterrows():
        print(f"  {r['rank_in_live_day']:>4.0f} {r['ts_code']:<12} {str(r.get('name','')):<8} s={r['overnight_live_score']:.4f} pass={r['live_pass_risk_filter']}")


if __name__ == "__main__":
    main()
