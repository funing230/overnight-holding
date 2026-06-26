#!/usr/bin/env python3
"""One-command multistage overnight live workflow.

Pipeline:
1. Deterministic recall from pre-close snapshot: Top50.
2. Heavy TradingAgents2.0 review: Top50 -> Top15.
3. Light TradingAgents2.0 fast review: Top15 -> structured scores.
4. Final fusion on final snapshot: deterministic + heavy + light -> Top3/Top5.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from config.default_config import DEFAULT_CONFIG
from dataflows.overnight_live_provider import run_live_inference
from dataflows.overnight_live_heavy_review_provider import build_heavy_review_payload, write_heavy_review_artifacts
from dataflows.overnight_live_review_provider import (
    build_live_review_payload,
    load_live_candidate_pool,
    neutral_review_scores,
    summarize_live_candidates,
    write_review_artifacts,
)
from dataflows.overnight_news_social_context import (
    build_news_social_context,
    summarize_news_social_context,
    NewsContextResult,
)
from dataflows.openclaw_context_provider import resolve_openclaw_context, build_openclaw_feature_frame
from dataflows.ashare_enrichment_provider import build_ashare_enrichment_features
from llm.pool import LLMPool
from dataflows.risk_veto_provider import build_risk_veto, RiskVetoPool

DEFAULT_OUT_ROOT = Path("data/overnight_live_multistage")

HEAVY_SYSTEM = """你是 TradingAgents2.0 的重度 pre-close research review agent。把 deterministic Top50 一夜持股法候选池压缩为 Top15 研究池，不直接给最终 Top5。只基于输入字段，不调用外部数据，不编造新闻/公告/财报。必须优先输出 HEAVY_REVIEW_JSON_START / HEAVY_REVIEW_JSON_END 包裹的 JSON。"""
FAST_SYSTEM = """你是 TradingAgents2.0 的轻量化 pre-close review agent。快速审查 heavy Top15 研究池，输出 AGENT_REVIEW_JSON_START / AGENT_REVIEW_JSON_END 包裹的 JSON。只基于输入字段，不调用外部数据，不编造新闻/公告/财报。"""


def _fmt_date(value: str) -> str:
    return str(value).replace("-", "")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _content_text(result: Any) -> str:
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            parts.append(str(item.get("text") or item.get("content") or "") if isinstance(item, dict) else str(item))
        return "\n".join([p for p in parts if p])
    return str(content)


def _snapshot_time_hint(snapshot_csv: str | Path | None) -> str | None:
    if not snapshot_csv:
        return None
    try:
        df = pd.read_csv(snapshot_csv, usecols=lambda c: c == "quote_time")
        return None if df.empty or "quote_time" not in df.columns else str(df["quote_time"].dropna().max())
    except Exception:
        return None


def _write_inference_outputs(result: dict[str, Any], out_dir: Path, top_n: int, pool_size: int, stage: str) -> dict[str, str]:
    _ensure_dir(out_dir)
    suffix = f"{_fmt_date(result['trade_date'])}_{stage}_top{top_n}_pool{pool_size}"
    paths = {
        "features": out_dir / f"live_features_{suffix}.csv",
        "scored": out_dir / f"live_scored_{suffix}.csv",
        "candidate_pool": out_dir / f"live_candidate_pool_{suffix}.csv",
        "selected": out_dir / f"live_selected_{suffix}.csv",
        "summary": out_dir / f"live_summary_{suffix}.md",
    }
    result["features"].to_csv(paths["features"], index=False)
    result["scored"].to_csv(paths["scored"], index=False)
    result["candidate_pool"].to_csv(paths["candidate_pool"], index=False)
    result["selected"].to_csv(paths["selected"], index=False)
    cols = [c for c in ["rank_in_live_day", "ts_code", "name", "name_x", "industry", "market", "overnight_live_score", "final_live_score", "heavy_score", "heavy_tier", "heavy_veto", "agent_score", "agent_risk_level", "agent_veto", "hot_mention_count", "social_bonus_score", "theme_names", "theme_bonus_score", "last_price", "quote_time", "live_return_vs_pre_close", "from_day_high", "live_range_pos", "live_reject_reasons"] if c in result["selected"].columns]
    lines = [f"# {stage} Live Inference Summary", "", f"- trade_date: `{result['trade_date']}`", f"- snapshot_csv: `{result['snapshot_csv']}`", f"- top_n: `{top_n}`", f"- candidate_pool_size: `{pool_size}`", "- exit_rule: `next_open_sell`", "- future_label_usage: `none`", "", "## Selected", ""]
    lines.append("- No selected candidates." if result["selected"].empty else result["selected"][cols].to_markdown(index=False))
    paths["summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {k: str(v) for k, v in paths.items()}


def _llm(role: str, model_key: str | None, mode: str):
    pool = LLMPool(DEFAULT_CONFIG.copy())
    if model_key:
        return pool.get_llm_by_key(model_key, mode=mode), f"{model_key}:{mode}"
    return pool.get_llm(role), f"role:{role}"


def _run_heavy(args, candidate_pool_csv: str, out_dir: Path, snapshot_hint: str | None, news_social_context=None, openclaw_context=None, risk_veto: RiskVetoPool | None = None) -> dict[str, Any]:
    started = time.time()
    pool = load_live_candidate_pool(candidate_pool_csv).head(args.heavy_top_k).copy()
    payload = build_heavy_review_payload(pool, args.trade_date, args.heavy_top_k, args.heavy_target_top_n, snapshot_hint, news_social_context=news_social_context, openclaw_context=openclaw_context)
    # ── Inject risk veto signals into heavy review prompt ──
    if risk_veto and risk_veto.vetoes:
        hard = [v for v in risk_veto.vetoes if v.severity == "hard"]
        soft = [v for v in risk_veto.vetoes if v.severity == "soft"]
        lines = ["\n## ⚠️ KNOWN RISK VETO SIGNALS (AUTO-DETECTED)\n"]
        if hard:
            lines.append(f"{len(hard)} stock(s) have HARD VETO (auto-reject):")
            for v in hard:
                lines.append(f"  - {v.ts_code}: {v.reason[:100]}")
        if soft:
            lines.append(f"\n{len(soft)} stock(s) have SOFT WARNING:")
            for v in soft:
                lines.append(f"  - {v.ts_code}: {v.reason[:100]}")
        lines.append("\nHard-veto stocks should be assigned heavy_tier='reject' with heavy_veto=true.")
        payload += "\n" + "\n".join(lines)
    _ensure_dir(out_dir)
    (out_dir / "heavy_review_prompt.md").write_text(payload + "\n", encoding="utf-8")
    if args.dry_run_heavy:
        reviews = [{"ts_code": r["ts_code"], "heavy_score": 0.5, "heavy_tier": "watch", "heavy_veto": False, "heavy_adjustment": 0.0, "heavy_keep_rank": i + 1, "heavy_reason": "dry_run_heavy_neutral", "heavy_risk_flags": []} for i, r in pool.reset_index(drop=True).iterrows()]
        decision = "HEAVY_REVIEW_JSON_START\n" + json.dumps({"trade_date": args.trade_date, "target_top_n": args.heavy_target_top_n, "reviews": reviews}, ensure_ascii=False, indent=2) + "\nHEAVY_REVIEW_JSON_END"
        llm_label = "dry_run_heavy_neutral"
    else:
        model, llm_label = _llm(args.heavy_role, args.heavy_model_key, args.heavy_mode)
        decision = _content_text(model.invoke([SystemMessage(content=HEAVY_SYSTEM), HumanMessage(content=payload)]))
    state = {"mode": "heavy_top50_to_top15_review", "trade_date": args.trade_date, "llm": llm_label, "candidate_count": int(len(pool)), "target_top_n": args.heavy_target_top_n, "snapshot_time_hint": snapshot_hint, "summary": summarize_live_candidates(pool, args.heavy_top_k), "news_social_context": summarize_news_social_context(news_social_context) if news_social_context is not None else None, "openclaw_context": None if openclaw_context is None else openclaw_context.get("summary"), "elapsed_seconds": round(time.time() - started, 3)}
    paths = write_heavy_review_artifacts(out_dir, state, pool, decision, target_top_n=args.heavy_target_top_n)
    return {"state": state, "paths": paths}


def _run_light(args, candidate_pool_csv: str, out_dir: Path, snapshot_hint: str | None, news_social_context=None, openclaw_context=None) -> dict[str, Any]:
    started = time.time()
    pool = load_live_candidate_pool(candidate_pool_csv).head(args.light_top_k).copy()
    light_news_context = news_social_context if args.light_include_news_social_context else None
    payload = build_live_review_payload(pool, args.trade_date, args.light_top_k, args.final_top_n, snapshot_hint, news_social_context=light_news_context, openclaw_context=openclaw_context)
    _ensure_dir(out_dir)
    (out_dir / "fast_agent_review_prompt.md").write_text(payload + "\n", encoding="utf-8")
    if args.dry_run_light:
        scores = neutral_review_scores(pool, reason="dry_run_light_neutral")
        decision = "AGENT_REVIEW_JSON_START\n" + json.dumps({"trade_date": args.trade_date, "target_top_n": args.final_top_n, "reviews": json.loads(scores.to_json(orient="records", force_ascii=False))}, ensure_ascii=False, indent=2) + "\nAGENT_REVIEW_JSON_END"
        llm_label = "dry_run_light_neutral"
    else:
        model, llm_label = _llm(args.light_role, args.light_model_key, args.light_mode)
        decision = _content_text(model.invoke([SystemMessage(content=FAST_SYSTEM), HumanMessage(content=payload)]))
    state = {"mode": "fast_one_shot_review_after_heavy", "trade_date": args.trade_date, "llm": llm_label, "candidate_count": int(len(pool)), "target_top_n": args.final_top_n, "snapshot_time_hint": snapshot_hint, "summary": summarize_live_candidates(pool, args.light_top_k), "news_social_context": summarize_news_social_context(light_news_context) if light_news_context is not None else None, "openclaw_context": None if openclaw_context is None else openclaw_context.get("summary"), "elapsed_seconds": round(time.time() - started, 3), "light_include_news_social_context": bool(args.light_include_news_social_context)}
    paths = write_review_artifacts(out_dir, state, pool, decision)
    return {"state": state, "paths": paths}


def _load_news_social_context_payload(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"News/social context JSON not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _news_context_from_payload(payload: dict[str, Any]) -> NewsContextResult:
    summary = payload.get("summary") or {}
    return NewsContextResult(
        news_top10=payload.get("news_top10", []),
        social_top10=payload.get("social_sentiment_top10", []),
        global_news_top10=payload.get("global_news_top10", []),
        vendor=str(summary.get("vendor", "")),
        notes=list(summary.get("notes", []) or []),
        social_hot_features=(payload.get("social_hot_context") or {}).get("features", []),
        social_hot_summary=(payload.get("social_hot_context") or {}).get("summary", {}),
        theme_hot_features=(payload.get("theme_hot_context") or {}).get("features", []),
        theme_hot_summary=(payload.get("theme_hot_context") or {}).get("summary", {}),
    )


def _maybe_resume_stage(paths: dict[str, str], required_keys: list[str]) -> bool:
    return all(Path(paths[k]).exists() for k in required_keys)


def _build_and_write_news_social_context(args, candidate_pool_csv: str, out_dir: Path, top_k: int) -> dict[str, Any]:
    started = time.time()
    _ensure_dir(out_dir)
    pool = load_live_candidate_pool(candidate_pool_csv).head(top_k).copy()
    openclaw_ctx = resolve_openclaw_context(
        args.trade_date,
        context_dir=args.openclaw_context_dir,
        root=args.openclaw_context_root,
    ) if not args.disable_openclaw_context else None
    if args.openclaw_context_required and (openclaw_ctx is None or not openclaw_ctx.loaded):
        reason = "disabled" if args.disable_openclaw_context else (openclaw_ctx.error if openclaw_ctx is not None else "not_loaded")
        raise RuntimeError(f"OpenClaw context required but unavailable: {reason}")
    ctx = build_news_social_context(
        pool,
        trade_date=args.trade_date,
        top_k_candidates=top_k,
        news_top_n=args.news_top_n,
        social_top_n=args.social_top_n,
        look_back_days=args.news_look_back_days,
        global_news_limit=args.global_news_top_n,
        enable_social_hot_context=(not args.disable_social_hot_context),
        social_hot_api_base=args.social_hot_api_base,
        social_hot_sources=[x.strip() for x in args.social_hot_sources.split(",") if x.strip()],
        social_hot_limit_per_source=args.social_hot_limit_per_source,
        enable_xueqiu=(not getattr(args, "disable_xueqiu", False)),
        enable_twitter=(not getattr(args, "disable_twitter", False)),
    )
    payload = {
        "trade_date": args.trade_date,
        "candidate_top_k": top_k,
        "summary": summarize_news_social_context(ctx),
        "news_top10": ctx.news_top10,
        "social_sentiment_top10": ctx.social_top10,
        "global_news_top10": ctx.global_news_top10,
        "social_hot_context": {
            "summary": ctx.social_hot_summary,
            "features": ctx.social_hot_features,
        },
        "theme_hot_context": {
            "summary": ctx.theme_hot_summary,
            "features": ctx.theme_hot_features,
        },
        "openclaw_context": None if openclaw_ctx is None else {
            "loaded": bool(openclaw_ctx.loaded),
            "context_dir": openclaw_ctx.context_dir,
            "manifest_path": openclaw_ctx.manifest_path,
            "summary": openclaw_ctx.summary,
            "payload": openclaw_ctx.payload,
            "prompt_block": openclaw_ctx.prompt_block,
            "error": openclaw_ctx.error,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    out_path = out_dir / "news_social_context.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    social_hot_features_path = out_dir / "social_hot_features.csv"
    pd.DataFrame(ctx.social_hot_features).drop(columns=["matched_hot_items"], errors="ignore").to_csv(social_hot_features_path, index=False)
    theme_hot_features_path = out_dir / "theme_hot_features.csv"
    pd.DataFrame(ctx.theme_hot_features).drop(columns=["matched_theme_items"], errors="ignore").to_csv(theme_hot_features_path, index=False)
    openclaw_features_path = None
    if openclaw_ctx is not None and openclaw_ctx.loaded:
        openclaw_features_path = out_dir / "openclaw_features.csv"
        build_openclaw_feature_frame(openclaw_ctx.payload).to_csv(openclaw_features_path, index=False)
    xueqiu_features_path = None
    if ctx.xueqiu_hot_features:
        xueqiu_features_path = out_dir / "xueqiu_hot_features.csv"
        pd.DataFrame(ctx.xueqiu_hot_features).to_csv(xueqiu_features_path, index=False)
    twitter_features_path = None
    if ctx.twitter_features:
        twitter_features_path = out_dir / "twitter_features.csv"
        pd.DataFrame(ctx.twitter_features).to_csv(twitter_features_path, index=False)
    return {
        "context": ctx,
        "path": str(out_path),
        "summary": payload["summary"],
        "social_hot_features_path": str(social_hot_features_path),
        "theme_hot_features_path": str(theme_hot_features_path),
        "openclaw_features_path": None if openclaw_features_path is None else str(openclaw_features_path),
        "xueqiu_hot_features_path": None if xueqiu_features_path is None else str(xueqiu_features_path),
        "twitter_features_path": None if twitter_features_path is None else str(twitter_features_path),
        "openclaw_context": None if openclaw_ctx is None else {
            "loaded": bool(openclaw_ctx.loaded),
            "context_dir": openclaw_ctx.context_dir,
            "manifest_path": openclaw_ctx.manifest_path,
            "summary": openclaw_ctx.summary,
            "payload": openclaw_ctx.payload,
            "prompt_block": openclaw_ctx.prompt_block,
            "error": openclaw_ctx.error,
        },
    }


def _run_prefetch_after_recall(args, candidate_pool_csv: str, out_dir: Path) -> dict[str, Any]:
    started = time.time()
    _ensure_dir(out_dir)
    cmd = [
        "python3",
        "scripts/prefetch_overnight_minute_cache.py",
        "--trade-date", args.trade_date,
        "--candidate-csv", candidate_pool_csv,
        "--symbol-column", "ts_code",
        "--rank-column", "rank_in_live_day",
        "--candidate-limit", str(args.minute_prefetch_candidate_limit),
        "--start-time", args.minute_prefetch_start_time,
        "--end-time", args.minute_prefetch_end_time,
        "--freq", args.minute_prefetch_freq,
        "--cache-dir", args.minute_cache_dir,
        "--out-root", str(out_dir),
    ]
    if args.minute_prefetch_missing_only:
        cmd.append("--missing-only")
    if args.minute_prefetch_force_refresh:
        cmd.append("--force-refresh")
    if args.minute_prefetch_max_symbols > 0:
        cmd += ["--max-symbols", str(args.minute_prefetch_max_symbols)]
    if args.minute_prefetch_shard_count > 1:
        cmd += ["--shard-id", str(args.minute_prefetch_shard_id), "--shard-count", str(args.minute_prefetch_shard_count)]

    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    state = {
        "enabled": True,
        "command": cmd,
        "returncode": int(proc.returncode),
        "elapsed_seconds": round(time.time() - started, 3),
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-4000:] if proc.stderr else "",
    }
    state_path = out_dir / "minute_prefetch_state.json"
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if proc.returncode != 0:
        # Minute prefetch is an opportunistic latency/feature enhancement.
        # In the 14:30-14:57 live window, upstream minute APIs may rate-limit;
        # do not abort the full Top5 workflow just because cache warm-up failed.
        # Downstream stages must proceed with available snapshot/history features
        # and carry this state in the manifest as degraded evidence.
        state["degraded"] = True
        state["degrade_reason"] = f"minute_prefetch_failed_returncode_{proc.returncode}"
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"state": state, "state_path": str(state_path)}


def _build_and_write_ashare_enrichment(args, candidate_pool_csv: str, out_dir: Path, top_k: int) -> dict[str, Any]:
    started = time.time()
    _ensure_dir(out_dir)
    pool = load_live_candidate_pool(candidate_pool_csv).head(top_k).copy()
    if "ts_code" not in pool.columns:
        raise ValueError(f"Candidate pool missing ts_code: {candidate_pool_csv}")
    ts_codes = pool["ts_code"].dropna().astype(str).tolist()
    features, manifest = build_ashare_enrichment_features(
        ts_codes,
        trade_date=args.trade_date,
        include_research=(not args.disable_ashare_research),
        include_business=(not args.disable_ashare_business),
    )
    features_path = out_dir / "ashare_enrichment_features.csv"
    features.to_csv(features_path, index=False)
    manifest = {
        **manifest,
        "enabled": True,
        "candidate_top_k": int(top_k),
        "features_path": str(features_path),
        "elapsed_seconds_total": round(time.time() - started, 3),
    }
    manifest_path = out_dir / "ashare_enrichment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return {"features_path": str(features_path), "manifest_path": str(manifest_path), "summary": manifest}


def main() -> None:
    p = argparse.ArgumentParser(description="Run multistage overnight live workflow: Top50 -> heavy Top15 -> light -> final TopN")
    p.add_argument("--trade-date", required=True)
    p.add_argument("--prefilter-snapshot-csv", required=True, help="Early/pre-close snapshot for Top50 recall")
    p.add_argument("--final-snapshot-csv", default=None, help="Final 14:55 snapshot; defaults to prefilter snapshot for smoke tests")
    p.add_argument("--history-feature-table", default=DEFAULT_CONFIG["overnight_feature_table_path"])
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--heavy-top-k", type=int, default=50)
    p.add_argument("--heavy-target-top-n", type=int, default=15)
    p.add_argument("--light-top-k", type=int, default=15)
    p.add_argument("--final-top-n", type=int, default=5)
    p.add_argument("--final-candidate-pool-size", type=int, default=50)
    p.add_argument("--live-weight", type=float, default=0.60)
    p.add_argument("--heavy-weight", type=float, default=0.25)
    p.add_argument("--light-weight", type=float, default=0.15)
    p.add_argument("--heavy-role", default="research_manager")
    p.add_argument("--heavy-model-key", default=None)
    p.add_argument("--heavy-mode", default="deepthink")
    p.add_argument("--light-role", default="market_analyst")
    p.add_argument("--light-model-key", default=None)
    p.add_argument("--light-mode", default="chat")
    p.add_argument("--dry-run-heavy", action="store_true")
    p.add_argument("--dry-run-light", action="store_true")
    p.add_argument("--enable-minute-prefetch", action="store_true", help="Run minute prefetch immediately after recall Top50 so minute API load is moved earlier than decision time")
    p.add_argument("--minute-cache-dir", default="data/overnight_mvp/cache/minute_1430_features")
    p.add_argument("--minute-prefetch-candidate-limit", type=int, default=50)
    p.add_argument("--minute-prefetch-start-time", default="14:30:00")
    p.add_argument("--minute-prefetch-end-time", default="15:00:00")
    p.add_argument("--minute-prefetch-freq", default="5min")
    p.add_argument("--minute-prefetch-missing-only", action="store_true")
    p.add_argument("--minute-prefetch-force-refresh", action="store_true")
    p.add_argument("--minute-prefetch-max-symbols", type=int, default=0)
    p.add_argument("--minute-prefetch-shard-id", type=int, default=0)
    p.add_argument("--minute-prefetch-shard-count", type=int, default=1)
    p.add_argument("--disable-xueqiu", action="store_true", help="Disable Xueqiu (雪球) hot stocks/posts context")
    p.add_argument("--disable-twitter", action="store_true", help="Disable Twitter/X sentiment context")
    p.add_argument("--news-top-n", type=int, default=10)
    p.add_argument("--social-top-n", type=int, default=10)
    p.add_argument("--global-news-top-n", type=int, default=10)
    p.add_argument("--news-look-back-days", type=int, default=3)
    p.add_argument("--disable-social-hot-context", action="store_true")
    p.add_argument("--social-hot-api-base", default=None)
    p.add_argument("--social-hot-sources", default="weibo,zhihu")
    p.add_argument("--social-hot-limit-per-source", type=int, default=20)
    p.add_argument("--disable-openclaw-context", action="store_true")
    p.add_argument("--light-include-news-social-context", action="store_true", help="Include full news/social context block in light prompt; disabled by default to reduce prompt size and timeout risk")
    p.add_argument("--resume-run-dir", default=None, help="Resume from an existing multistage run directory and skip stages whose artifacts already exist")
    p.add_argument("--openclaw-context-dir", default=None)
    p.add_argument("--openclaw-context-root", default="data/openclaw_context")
    p.add_argument("--openclaw-context-required", action="store_true")
    p.add_argument("--enable-ashare-enrichment", action="store_true", help="Fetch/merge A-share V3.1 enrichment features: fund flow, minute, LHB, institution seats, block trade, reports, business text")
    p.add_argument("--ashare-enrichment-top-k", type=int, default=50, help="Top-K recall candidates to enrich")
    p.add_argument("--disable-ashare-research", action="store_true", help="Skip Eastmoney research report enrichment")
    p.add_argument("--disable-ashare-business", action="store_true", help="Skip THS business/company profile enrichment")
    args = p.parse_args()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir)
        run_ts = run_dir.name
    else:
        run_dir = Path(args.out_root) / args.trade_date / run_ts
    recall_dir, minute_prefetch_dir, news_social_dir, ashare_enrich_dir, heavy_dir, light_dir, final_dir = (
        run_dir / "01_recall_top50",
        run_dir / "01b_minute_prefetch",
        run_dir / "01c_news_social_top10",
        run_dir / "01d_ashare_enrichment",
        run_dir / "02_heavy_review",
        run_dir / "03_light_review",
        run_dir / "04_final_fusion",
    )

    pre_hint = _snapshot_time_hint(args.prefilter_snapshot_csv)
    final_snapshot = args.final_snapshot_csv or args.prefilter_snapshot_csv
    final_hint = _snapshot_time_hint(final_snapshot)

    recall_candidate_pool_path = recall_dir / f"live_candidate_pool_{_fmt_date(args.trade_date)}_recall_top{args.heavy_top_k}_pool{args.heavy_top_k}.csv"
    recall_paths = {
        "features": str(recall_dir / f"live_features_{_fmt_date(args.trade_date)}_recall_top{args.heavy_top_k}_pool{args.heavy_top_k}.csv"),
        "scored": str(recall_dir / f"live_scored_{_fmt_date(args.trade_date)}_recall_top{args.heavy_top_k}_pool{args.heavy_top_k}.csv"),
        "candidate_pool": str(recall_candidate_pool_path),
        "selected": str(recall_dir / f"live_selected_{_fmt_date(args.trade_date)}_recall_top{args.heavy_top_k}_pool{args.heavy_top_k}.csv"),
        "summary": str(recall_dir / f"live_summary_{_fmt_date(args.trade_date)}_recall_top{args.heavy_top_k}_pool{args.heavy_top_k}.md"),
    }
    if args.resume_run_dir and _maybe_resume_stage(recall_paths, ["candidate_pool", "scored", "selected"]):
        recall = {
            "trade_date": args.trade_date,
            "snapshot_csv": args.prefilter_snapshot_csv,
            "candidate_pool": pd.read_csv(recall_paths["candidate_pool"]),
            "selected": pd.read_csv(recall_paths["selected"]),
        }
    else:
        recall = run_live_inference(args.prefilter_snapshot_csv, args.trade_date, args.history_feature_table, top_n=args.heavy_top_k, candidate_pool_size=args.heavy_top_k)
        recall_paths = _write_inference_outputs(recall, recall_dir, args.heavy_top_k, args.heavy_top_k, "recall")

    minute_prefetch = None
    minute_state_path = minute_prefetch_dir / "minute_prefetch_state.json"
    if args.enable_minute_prefetch:
        if args.resume_run_dir and minute_state_path.exists():
            minute_prefetch = json.loads(minute_state_path.read_text(encoding="utf-8"))
            minute_prefetch = {"state": minute_prefetch, "state_path": str(minute_state_path)}
        else:
            minute_prefetch = _run_prefetch_after_recall(args, recall_paths["candidate_pool"], minute_prefetch_dir)

    news_social_path = news_social_dir / "news_social_context.json"
    if args.resume_run_dir and news_social_path.exists():
        news_payload = _load_news_social_context_payload(news_social_path)
        news_social = {
            "context": _news_context_from_payload(news_payload),
            "path": str(news_social_path),
            "summary": news_payload.get("summary"),
            "social_hot_features_path": str(news_social_dir / "social_hot_features.csv") if (news_social_dir / "social_hot_features.csv").exists() else None,
            "theme_hot_features_path": str(news_social_dir / "theme_hot_features.csv") if (news_social_dir / "theme_hot_features.csv").exists() else None,
            "openclaw_features_path": str(news_social_dir / "openclaw_features.csv") if (news_social_dir / "openclaw_features.csv").exists() else None,
            "openclaw_context": news_payload.get("openclaw_context"),
        }
    else:
        news_social = _build_and_write_news_social_context(args, recall_paths["candidate_pool"], news_social_dir, top_k=args.heavy_top_k)

    ashare_enrichment = None
    ashare_features_path = ashare_enrich_dir / "ashare_enrichment_features.csv"
    ashare_manifest_path = ashare_enrich_dir / "ashare_enrichment_manifest.json"
    if args.enable_ashare_enrichment:
        if args.resume_run_dir and ashare_features_path.exists() and ashare_manifest_path.exists():
            ashare_enrichment = {
                "features_path": str(ashare_features_path),
                "manifest_path": str(ashare_manifest_path),
                "summary": json.loads(ashare_manifest_path.read_text(encoding="utf-8")),
            }
        else:
            ashare_enrichment = _build_and_write_ashare_enrichment(args, recall_paths["candidate_pool"], ashare_enrich_dir, top_k=args.ashare_enrichment_top_k)

    heavy_paths = {
        "heavy_review_report": str(heavy_dir / "heavy_review_report.md"),
        "heavy_review_scores": str(heavy_dir / "heavy_review_scores.csv"),
        "heavy_review_state": str(heavy_dir / "heavy_review_state.json"),
        "heavy_review_parse_manifest": str(heavy_dir / "heavy_review_parse_manifest.json"),
        "heavy_selected_top15": str(heavy_dir / "heavy_selected_top15.csv"),
        "heavy_review_prompt": str(heavy_dir / "heavy_review_prompt.md"),
    }
    if args.resume_run_dir and _maybe_resume_stage(heavy_paths, ["heavy_review_scores", "heavy_selected_top15", "heavy_review_state"]):
        heavy_state = json.loads(Path(heavy_paths["heavy_review_state"]).read_text(encoding="utf-8"))
        heavy = {"state": heavy_state, "paths": heavy_paths}
    else:
        # ── Risk veto scan: check candidates for fundamental risks ──
        risk_veto_pool = build_risk_veto(pd.read_csv(recall_paths["candidate_pool"]), trade_date=args.trade_date)
        if risk_veto_pool.hard_veto_count > 0 or risk_veto_pool.soft_warning_count > 0:
            print(f"Risk veto: {risk_veto_pool.hard_veto_count} hard, {risk_veto_pool.soft_warning_count} soft")
        heavy = _run_heavy(args, recall_paths["candidate_pool"], heavy_dir, pre_hint, news_social_context=news_social["context"], openclaw_context=news_social.get("openclaw_context"), risk_veto=risk_veto_pool)

    light_paths = {
        "agent_review_report": str(light_dir / "agent_review_report.md"),
        "agent_review_scores": str(light_dir / "agent_review_scores.csv"),
        "agent_review_state": str(light_dir / "agent_review_state.json"),
        "agent_review_parse_manifest": str(light_dir / "agent_review_parse_manifest.json"),
    }
    if args.resume_run_dir and _maybe_resume_stage(light_paths, ["agent_review_scores", "agent_review_state"]):
        light_state = json.loads(Path(light_paths["agent_review_state"]).read_text(encoding="utf-8"))
        light = {"state": light_state, "paths": light_paths}
    else:
        light = _run_light(args, heavy["paths"]["heavy_selected_top15"], light_dir, pre_hint, news_social_context=news_social["context"], openclaw_context=news_social.get("openclaw_context"))

    final_paths_guess = {
        "features": str(final_dir / f"live_features_{_fmt_date(args.trade_date)}_final_top{args.final_top_n}_pool{args.final_candidate_pool_size}.csv"),
        "scored": str(final_dir / f"live_scored_{_fmt_date(args.trade_date)}_final_top{args.final_top_n}_pool{args.final_candidate_pool_size}.csv"),
        "candidate_pool": str(final_dir / f"live_candidate_pool_{_fmt_date(args.trade_date)}_final_top{args.final_top_n}_pool{args.final_candidate_pool_size}.csv"),
        "selected": str(final_dir / f"live_selected_{_fmt_date(args.trade_date)}_final_top{args.final_top_n}_pool{args.final_candidate_pool_size}.csv"),
        "summary": str(final_dir / f"live_summary_{_fmt_date(args.trade_date)}_final_top{args.final_top_n}_pool{args.final_candidate_pool_size}.md"),
    }
    if args.resume_run_dir and _maybe_resume_stage(final_paths_guess, ["selected", "scored", "candidate_pool"]):
        final = {"selected": pd.read_csv(final_paths_guess["selected"])}
        final_paths = final_paths_guess
    else:
        final = run_live_inference(
            final_snapshot,
            args.trade_date,
            args.history_feature_table,
            top_n=args.final_top_n,
            candidate_pool_size=args.final_candidate_pool_size,
            heavy_review_scores_path=heavy["paths"]["heavy_review_scores"],
            light_review_scores_path=light["paths"]["agent_review_scores"],
            social_hot_features_path=news_social.get("social_hot_features_path"),
            theme_hot_features_path=news_social.get("theme_hot_features_path"),
            openclaw_features_path=news_social.get("openclaw_features_path"),
            xueqiu_hot_features_path=news_social.get("xueqiu_hot_features_path"),
            twitter_features_path=news_social.get("twitter_features_path"),
            ashare_enrichment_features_path=None if ashare_enrichment is None else ashare_enrichment.get("features_path"),
            live_weight=args.live_weight,
            heavy_weight=args.heavy_weight,
            light_weight=args.light_weight,
        )
        final_paths = _write_inference_outputs(final, final_dir, args.final_top_n, args.final_candidate_pool_size, "final")

    manifest = {"trade_date": args.trade_date, "run_ts": run_ts, "resume_run_dir": args.resume_run_dir, "prefilter_snapshot_csv": args.prefilter_snapshot_csv, "final_snapshot_csv": final_snapshot, "prefilter_snapshot_time_hint": pre_hint, "final_snapshot_time_hint": final_hint, "weights": {"live_weight": args.live_weight, "heavy_weight": args.heavy_weight, "light_weight": args.light_weight}, "light_include_news_social_context": bool(args.light_include_news_social_context), "stages": {"recall": recall_paths, "minute_prefetch": minute_prefetch, "news_social": {"path": news_social["path"], "summary": news_social["summary"], "social_hot_features_path": news_social.get("social_hot_features_path"), "theme_hot_features_path": news_social.get("theme_hot_features_path"), "openclaw_features_path": news_social.get("openclaw_features_path"), "openclaw_context": news_social.get("openclaw_context")}, "ashare_enrichment": ashare_enrichment, "heavy": heavy, "light": light, "final": final_paths}, "rows": {"recall_candidate_pool": int(len(pd.read_csv(recall_paths["candidate_pool"]))), "heavy_selected_top15": int(len(pd.read_csv(heavy["paths"]["heavy_selected_top15"]))), "final_selected": int(len(pd.read_csv(final_paths["selected"])))}}
    manifest_path = run_dir / "multistage_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"Wrote multistage manifest: {manifest_path}")
    print(f"Recall candidate pool: {recall_paths['candidate_pool']}")
    if minute_prefetch:
        print(f"Minute prefetch state: {minute_prefetch['state_path']}")
    print(f"News/social context: {news_social['path']}")
    if news_social.get("social_hot_features_path"):
        print(f"Social hot features: {news_social['social_hot_features_path']}")
    if news_social.get("theme_hot_features_path"):
        print(f"Theme hot features: {news_social['theme_hot_features_path']}")
    if news_social.get("openclaw_features_path"):
        print(f"OpenClaw features: {news_social['openclaw_features_path']}")
    if ashare_enrichment:
        print(f"A-share enrichment features: {ashare_enrichment['features_path']}")
    print(f"Heavy scores: {heavy['paths']['heavy_review_scores']}")
    print(f"Heavy Top15: {heavy['paths']['heavy_selected_top15']}")
    print(f"Light scores: {light['paths']['agent_review_scores']}")
    print(f"Final selected: {final_paths['selected']}")


if __name__ == "__main__":
    main()
