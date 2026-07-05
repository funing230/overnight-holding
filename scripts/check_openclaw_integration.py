#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dataflows.openclaw_context_provider import (
    build_openclaw_feature_frame,
    resolve_openclaw_context,
)
from dataflows.overnight_live_selector_review_provider import build_selector_review_payload
from dataflows.overnight_live_provider import apply_multi_stage_review_fusion
from dataflows.overnight_live_scorer_review_provider import build_scorer_review_payload, load_live_candidate_pool
from dataflows.overnight_news_social_context import NewsContextResult


WORKDIR = Path(__file__).resolve().parents[1]
DEFAULT_TRADE_DATE = "2026-05-20"
DEFAULT_CONTEXT_ROOT = WORKDIR / "data/openclaw_context"
DEFAULT_POOL_CSV = WORKDIR / "data/overnight_live_multistage_step1_test/2026-05-20/20260520_160543/01_recall_top50/live_candidate_pool_20260520_recall_top10_pool10.csv"
DEFAULT_OUT_DIR = WORKDIR / "data/openclaw_smoke_check"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)

    ctx = resolve_openclaw_context(DEFAULT_TRADE_DATE, root=DEFAULT_CONTEXT_ROOT)
    _assert(ctx.loaded, f"OpenClaw context not loaded: {ctx.error}")
    _assert(bool(ctx.payload), "OpenClaw payload is empty")
    _assert("macro_news_top" in ctx.prompt_block, "OpenClaw prompt block missing macro_news_top")
    _assert("ticker_event_top" in ctx.prompt_block, "OpenClaw prompt block missing ticker_event_top")

    feature_df = build_openclaw_feature_frame(ctx.payload)
    _assert(not feature_df.empty, "OpenClaw feature frame is empty")
    _assert("ts_code" in feature_df.columns, "OpenClaw feature frame missing ts_code")
    feature_path = DEFAULT_OUT_DIR / "openclaw_features.csv"
    feature_df.to_csv(feature_path, index=False)

    pool = load_live_candidate_pool(DEFAULT_POOL_CSV).head(5)
    news_ctx = NewsContextResult(
        vendor="smoke-check",
        notes=[],
        news_top10=[],
        social_top10=[],
        global_news_top10=[],
        social_hot_summary={},
        social_hot_features=[],
        theme_hot_summary={},
        theme_hot_features=[],
    )
    oc = {"summary": ctx.summary, "payload": ctx.payload}
    selector_payload = build_selector_review_payload(pool, DEFAULT_TRADE_DATE, top_k=5, target_top_n=3, snapshot_time_hint="14:35", news_social_context=news_ctx, openclaw_context=oc)
    scorer_payload = build_scorer_review_payload(pool, DEFAULT_TRADE_DATE, top_k=5, target_top_n=3, snapshot_time_hint="14:35", news_social_context=news_ctx, openclaw_context=oc)
    _assert("\"openclaw_context\"" in heavy_payload, "Selector payload summary missing openclaw_context")
    _assert("macro_news_top" in heavy_payload and "ticker_event_top" in heavy_payload, "Selector payload missing OpenClaw prompt block")
    _assert("\"openclaw_context\"" in light_payload, "Scorer payload summary missing openclaw_context")
    _assert("macro_news_top" in light_payload and "ticker_event_top" in light_payload, "Scorer payload missing OpenClaw prompt block")

    base = pd.DataFrame([
        {"ts_code": "600009.SH", "overnight_live_score": 0.60, "live_pass_risk_filter": True, "live_reject_reasons": ""},
        {"ts_code": "002049.SZ", "overnight_live_score": 0.59, "live_pass_risk_filter": True, "live_reject_reasons": ""},
        {"ts_code": "600309.SH", "overnight_live_score": 0.61, "live_pass_risk_filter": True, "live_reject_reasons": ""},
    ])
    no_oc = apply_multi_stage_review_fusion(base, live_weight=1.0, heavy_weight=0.0, light_weight=0.0)
    with_oc = apply_multi_stage_review_fusion(base, openclaw_features_path=feature_path, live_weight=1.0, heavy_weight=0.0, light_weight=0.0)
    merged = no_oc[["ts_code", "final_live_score"]].merge(
        with_oc[["ts_code", "final_live_score", "openclaw_feature_score", "openclaw_risk_penalty", "openclaw_theme_names", "openclaw_catalyst_summary"]],
        on="ts_code",
        suffixes=("_base", "_oc"),
    )
    merged["delta"] = merged["final_live_score_oc"] - merged["final_live_score_base"]
    delta_map = {row["ts_code"]: float(row["delta"]) for row in merged.to_dict(orient="records")}
    _assert(abs(delta_map.get("600009.SH", 0.0) - 0.08) < 1e-9, f"Unexpected delta for 600009.SH: {delta_map.get('600009.SH')}")
    _assert(abs(delta_map.get("002049.SZ", 0.0) - 0.0476) < 1e-9, f"Unexpected delta for 002049.SZ: {delta_map.get('002049.SZ')}")
    _assert(abs(delta_map.get("600309.SH", 1.0)) < 1e-9, f"Unexpected delta for 600309.SH: {delta_map.get('600309.SH')}")

    report = {
        "trade_date": DEFAULT_TRADE_DATE,
        "context_dir": ctx.context_dir,
        "feature_rows": int(len(feature_df)),
        "feature_columns": list(feature_df.columns),
        "delta_check": merged.to_dict(orient="records"),
        "selector_prompt_has_openclaw": True,
        "scorer_prompt_has_openclaw": True,
        "status": "ok",
    }
    report_path = DEFAULT_OUT_DIR / "openclaw_smoke_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "report_path": str(report_path), "feature_path": str(feature_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
