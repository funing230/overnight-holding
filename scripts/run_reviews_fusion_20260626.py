#!/usr/bin/env python3
"""Minimal Heavy → Light → Fusion pipeline. Bypasses broken y_finance imports."""

import json, sys, time, re
from pathlib import Path

import pandas as pd
import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.default_config import DEFAULT_CONFIG
from llm.pool import LLMPool
from dataflows.overnight_live_provider import (
    load_snapshot_csv, load_history_feature_table, latest_history_by_symbol,
    build_live_feature_frame, score_live_candidates, apply_live_risk_filters,
    apply_multi_stage_review_fusion, LiveOvernightConfig,
)

OUT_DIR = Path("data/output/20260626_weekend")
TOP50  = OUT_DIR / "live_candidate_pool_top50.csv"
HEAVY_DIR  = OUT_DIR / "selector_review"
LIGHT_DIR  = OUT_DIR / "scorer_review"
FUSION_DIR = OUT_DIR / "final_fusion"
for d in [HEAVY_DIR, LIGHT_DIR, FUSION_DIR]: d.mkdir(parents=True, exist_ok=True)

TRADE_DATE = "2026-06-26"
EXT_FEATURE = Path("data/overnight_mvp/features/overnight_features_ext_20260626.csv")
SNAPSHOT = Path("data/snapshots/live_snapshot_20260626.csv")

HEAVY_SYSTEM = "你是TradingAgents2.0的筛选审查Agent。把deterministic Top50一夜持股法候选池压缩为Top15研究池。只基于输入字段，不编造新闻/公告/财报。必须优先输出SELECTOR_REVIEW_JSON_START/SELECTOR_REVIEW_JSON_END包裹的JSON。"
SCORER_SYSTEM = "你是TradingAgents2.0的评分复核Agent。快速审查selector Top15研究池。输出SCORER_REVIEW_JSON_START/SCORER_REVIEW_JSON_END包裹的JSON。只基于输入字段，不编造新闻/公告/财报。"


def _content_text(result):
    c = getattr(result, "content", result)
    if isinstance(c, str): return c
    if isinstance(c, list):
        return "\n".join(str(x.get("text") or x.get("content") or "") if isinstance(x, dict) else str(x) for x in c)
    return str(c)


def _compact(rows, cols):
    """Create compact JSON from dataframe rows for prompt."""
    records = []
    for _, r in rows.iterrows():
        rec = {}
        for c in cols:
            if c in r.index:
                v = r[c]
                rec[c] = None if pd.isna(v) else (float(v) if isinstance(v, (np.float64, float)) and abs(float(v)) > 1e6 else v)
        records.append(rec)
    return records


def main():
    pool = pd.read_csv(TOP50)
    print(f"Top50 pool: {len(pool)} stocks\n")

    # ── Heavy Review ──
    print("=" * 60)
    print("HEAVY REVIEW: Top50 → Top15 (deepthink)")
    print("=" * 60)

    # Build compact prompt manually
    cols = ["rank_in_live_day", "ts_code", "name", "industry", "market", "overnight_live_score",
            "last_price", "live_return_vs_prev_close", "live_range_pos", "from_day_high",
            "hist_ret_close_1d", "hist_ret_close_3d", "hist_ret_close_5d",
            "hist_overnight_prev_1d", "hist_overnight_prev_3d_mean", "hist_overnight_positive_rate_5d",
            "live_pass_risk_filter"]
    cols = [c for c in cols if c in pool.columns]
    records = json.loads(pool[cols].head(50).to_json(orient="records", force_ascii=False))
    # Make numbers readable
    for rec in records:
        for k, v in list(rec.items()):
            if isinstance(v, float): rec[k] = round(v, 4)

    summary = {
        "candidate_count": len(pool),
        "scores": {"min": round(float(pool["overnight_live_score"].min()), 4),
                   "median": round(float(pool["overnight_live_score"].median()), 4),
                   "max": round(float(pool["overnight_live_score"].max()), 4)},
    }
    if "industry" in pool.columns:
        summary["industry_counts"] = pool["industry"].fillna("UNKNOWN").value_counts().to_dict()

    heavy_payload = f"""你是TradingAgents2.0的A股一夜持股法重度候选池研究器。

目标：把deterministic Top50候选池压缩成高质量Top15研究池。

交易规则：买入窗口=收盘前，卖出=次日开盘。禁止使用未来信息，不要编造新闻/公告/财报。
你可以基于行业集中、尾盘结构、波动异常、流动性、短期趋势质量、历史overnight稳定性给出保留/降级/veto。

trade_date: {TRADE_DATE}
target_top_n: 15

候选池摘要：
```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

候选池明细（50只）：
```json
{json.dumps(records, ensure_ascii=False, indent=2)}
```

请严格按以下格式输出：

SELECTOR_REVIEW_JSON_START
{{
  "trade_date": "{TRADE_DATE}",
  "target_top_n": 15,
  "top_picks": [
    {{
      "ts_code": "000001.SZ",
      "heavy_score": 0.85,
      "heavy_tier": "core|watch",
      "heavy_veto": false,
      "heavy_adjustment": 0.05,
      "heavy_keep_rank": 1,
      "heavy_reason": "一句话，<=20字",
      "heavy_risk_flags": []
    }}
  ],
  "rejects": [],
  "summary": {{"core_count": 0, "watch_top15_count": 0, "reject_count": 0, "notes": "<=120字"}}
}}
SELECTOR_REVIEW_JSON_END

不要输出长篇推理。top_picks最多15条。rejects只写明确应veto的。"""

    (HEAVY_DIR / "prompt.md").write_text(heavy_payload)
    print(f"Prompt: {len(heavy_payload)} chars")

    llm_pool = LLMPool(DEFAULT_CONFIG.copy())
    model = llm_pool.get_llm_by_key("yuanlan4", mode="deepthink")
    print(f"Heavy LLM: yuanlan4:deepthink")

    t0 = time.time()
    try:
        resp = model.invoke([SystemMessage(content=HEAVY_SYSTEM), HumanMessage(content=heavy_payload)])
        h_decision = _content_text(resp)
        print(f"  Response: {len(h_decision)} chars in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  LLM ERROR: {e} — using neutral fallback")
        h_decision = "SELECTOR_REVIEW_JSON_START\n" + json.dumps({
            "trade_date": TRADE_DATE, "target_top_n": 15,
            "top_picks": [{"ts_code": r["ts_code"], "heavy_score": 0.5, "heavy_tier": "watch",
                          "heavy_veto": False, "heavy_adjustment": 0.0,
                          "heavy_keep_rank": i+1, "heavy_reason": "neutral_fallback", "heavy_risk_flags": []}
                         for i, r in pool.head(15).reset_index(drop=True).iterrows()]
        }) + "\nSELECTOR_REVIEW_JSON_END"

    (HEAVY_DIR / "response.md").write_text(h_decision)

    # Parse heavy JSON
    m = re.search(r'SELECTOR_REVIEW_JSON_START\s*(\{.*?\})\s*SELECTOR_REVIEW_JSON_END', h_decision, re.S)
    if m:
        h_data = json.loads(m.group(1))
        top_picks = pd.DataFrame(h_data.get("top_picks", []))
        rejects = pd.DataFrame(h_data.get("rejects", []))
    else:
        print("  WARNING: No JSON found in heavy response!")
        top_picks = pd.DataFrame()
        rejects = pd.DataFrame()

    # Build heavy review CSV
    heavy_rows = []
    pool_codes = set(pool["ts_code"])
    assigned = set()

    for _, r in top_picks.iterrows():
        code = r.get("ts_code", "")
        heavy_rows.append({
            "ts_code": code, "heavy_score": r.get("heavy_score", 0.5),
            "heavy_tier": r.get("heavy_tier", "watch"),
            "heavy_veto": False, "heavy_adjustment": r.get("heavy_adjustment", 0.0),
            "heavy_keep_rank": int(r.get("heavy_keep_rank", len(heavy_rows)+1)),
            "heavy_reason": r.get("heavy_reason", ""), "heavy_risk_flags": "[]",
        })
        assigned.add(code)

    for _, r in rejects.iterrows():
        code = r.get("ts_code", "")
        heavy_rows.append({
            "ts_code": code, "heavy_score": r.get("heavy_score", 0.2),
            "heavy_tier": "reject", "heavy_veto": True,
            "heavy_adjustment": r.get("heavy_adjustment", -0.1),
            "heavy_keep_rank": 999, "heavy_reason": r.get("heavy_reason", "rejected"),
            "heavy_risk_flags": "[]",
        })
        assigned.add(code)

    # Fill unassigned with neutral
    for i, (_, r) in enumerate(pool.iterrows()):
        if r["ts_code"] not in assigned:
            heavy_rows.append({
                "ts_code": r["ts_code"], "heavy_score": 0.5, "heavy_tier": "watch",
                "heavy_veto": False, "heavy_adjustment": 0.0,
                "heavy_keep_rank": 100 + i, "heavy_reason": "unmentioned_neutral",
                "heavy_risk_flags": "[]",
            })

    heavy_df = pd.DataFrame(heavy_rows)
    # Deduplicate
    heavy_df = heavy_df.drop_duplicates("ts_code", keep="first")
    heavy_csv = HEAVY_DIR / "selector_review_scores.csv"
    heavy_df.to_csv(heavy_csv, index=False)
    print(f"  Heavy scores: {len(heavy_df)} rows → {heavy_csv}")

    # Show heavy results
    heavy_sorted = heavy_df.sort_values("heavy_keep_rank")
    print(f"\nHeavy Review results:")
    for _, r in heavy_sorted.head(20).iterrows():
        flag = "🟢" if r["heavy_tier"] == "core" else "🟡" if r["heavy_tier"] == "watch" else "🔴"
        print(f"  {flag} {r['ts_code']:<12} tier={r['heavy_tier']:<8} s={r['heavy_score']:.2f} veto={r['heavy_veto']} | {r.get('heavy_reason','')}")

    # ── Light Review ──
    print("\n" + "=" * 60)
    print("LIGHT REVIEW: Top15 → 5 (chat)")
    print("=" * 60)

    # Filter to heavy's top picks as the study pool
    study = heavy_sorted[heavy_sorted["heavy_tier"] != "reject"].head(15).copy()
    study = study.merge(pool[["ts_code", "name", "industry", "overnight_live_score",
                               "last_price", "live_return_vs_prev_close", "live_range_pos",
                               "from_day_high"]], on="ts_code", how="left", suffixes=("", "_p"))

    light_records = []
    for _, r in study.iterrows():
        lr = {}
        for c in ["ts_code", "name", "industry", "overnight_live_score", "last_price",
                   "live_return_vs_prev_close", "live_range_pos", "from_day_high",
                   "heavy_tier", "heavy_score", "heavy_reason"]:
            if c in r.index:
                v = r[c]
                lr[c] = round(float(v), 4) if isinstance(v, (float, np.float64)) and not pd.isna(v) else v
        light_records.append(lr)

    light_payload = f"""你是TradingAgents2.0的A股一夜持股法盘前尾盘缓冲区审查器。

目标：在14:55前对候选池做综合分析。给出每只股票的agent_score、风险等级、是否veto，以及排序调整建议。

交易规则：买入窗口=收盘前，卖出=次日开盘。禁止使用未来信息，不要编造新闻/公告/财报。

trade_date: {TRADE_DATE}
target_final_top_n: 5

候选池明细（{len(light_records)}只）：
```json
{json.dumps(light_records, ensure_ascii=False, indent=2)}
```

输出两部分：1. 人类可读分析 2. 机器可读JSON：

SCORER_REVIEW_JSON_START
{{
  "trade_date": "{TRADE_DATE}",
  "target_top_n": 5,
  "reviews": [
    {{
      "ts_code": "000001.SZ",
      "agent_score": 0.85,
      "agent_risk_level": "low|medium|high",
      "agent_veto": false,
      "agent_adjustment": 0.05,
      "agent_reason": "一句话"
    }}
  ]
}}
SCORER_REVIEW_JSON_END

每个候选都必须有reviews记录。veto=true的股票最终不能进入Top5。"""

    (LIGHT_DIR / "prompt.md").write_text(light_payload)
    print(f"Prompt: {len(light_payload)} chars")

    model2 = llm_pool.get_llm_by_key("yuanlan4", mode="chat")
    print(f"Light LLM: yuanlan4:chat")

    t0 = time.time()
    try:
        resp2 = model2.invoke([SystemMessage(content=SCORER_SYSTEM), HumanMessage(content=light_payload)])
        l_decision = _content_text(resp2)
        print(f"  Response: {len(l_decision)} chars in {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"  LLM ERROR: {e} — using neutral")
        l_decision = "SCORER_REVIEW_JSON_START\n" + json.dumps({
            "trade_date": TRADE_DATE, "target_top_n": 5,
            "reviews": [{"ts_code": r["ts_code"], "agent_score": 0.5, "agent_risk_level": "medium",
                        "agent_veto": False, "agent_adjustment": 0.0, "agent_reason": "neutral_fallback"}
                       for _, r in study.iterrows()]
        }) + "\nSCORER_REVIEW_JSON_END"

    (LIGHT_DIR / "response.md").write_text(l_decision)

    # Parse light JSON
    m2 = re.search(r'SCORER_REVIEW_JSON_START\s*(\{.*?\})\s*SCORER_REVIEW_JSON_END', l_decision, re.S)
    if m2:
        l_data = json.loads(m2.group(1))
        l_reviews = pd.DataFrame(l_data.get("reviews", []))
    else:
        print("  WARNING: No JSON in light response!")
        l_reviews = pd.DataFrame()

    # Build light review CSV
    light_rows = []
    study_codes = set(study["ts_code"])
    for _, r in l_reviews.iterrows():
        code = r.get("ts_code", "")
        light_rows.append({
            "ts_code": code, "agent_score": r.get("agent_score", 0.5),
            "agent_risk_level": r.get("agent_risk_level", "medium"),
            "agent_veto": str(r.get("agent_veto", False)).lower() in ("true", "1", "yes"),
            "agent_adjustment": r.get("agent_adjustment", 0.0),
            "agent_reason": r.get("agent_reason", ""),
        })
    # Fill missing with neutral
    for _, r in study.iterrows():
        if r["ts_code"] not in {x["ts_code"] for x in light_rows}:
            light_rows.append({
                "ts_code": r["ts_code"], "agent_score": 0.5, "agent_risk_level": "medium",
                "agent_veto": False, "agent_adjustment": 0.0, "agent_reason": "unmentioned_neutral",
            })

    light_df = pd.DataFrame(light_rows).drop_duplicates("ts_code", keep="first")
    light_csv = LIGHT_DIR / "scorer_review_scores.csv"
    light_df.to_csv(light_csv, index=False)
    print(f"\nLight Review scores → {light_csv}")

    for _, r in light_df.iterrows():
        flag = "✓" if not r["agent_veto"] else "✗"
        print(f"  {flag} {r['ts_code']:<12} s={r['agent_score']:.2f} risk={r['agent_risk_level']:<8} adj={r['agent_adjustment']:+.2f} | {r.get('agent_reason','')}")

    # ── Final Fusion ──
    print("\n" + "=" * 60)
    print("FINAL FUSION → Top5")
    print("=" * 60)

    snapshot = load_snapshot_csv(SNAPSHOT)
    history = load_history_feature_table(EXT_FEATURE)
    # Pass 2026-06-27 as "current" date so June 26 data becomes "history"
    latest = latest_history_by_symbol(history, trade_date="2026-06-27")
    features = build_live_feature_frame(snapshot, latest, "2026-06-27")
    scored = score_live_candidates(features)
    cfg = LiveOvernightConfig(history_feature_table_path=EXT_FEATURE)
    scored = apply_live_risk_filters(scored, cfg)

    fused = apply_multi_stage_review_fusion(
        scored,
        selector_review_scores_path=str(heavy_csv),
        scorer_review_scores_path=str(light_csv),
        live_weight=0.60, heavy_weight=0.25, light_weight=0.15,
    )

    top5 = fused[fused["live_pass_risk_filter"]].sort_values("final_live_score", ascending=False).head(5)

    print(f"\n╔{'═'*80}╗")
    print(f"║  🏆 FINAL TOP 5 — 周一(6/29) 一夜持股法买入候选  ║")
    print(f"╠{'═'*80}╣")
    for i, (_, r) in enumerate(top5.iterrows(), 1):
        f_score = r["final_live_score"]
        l_score = r["overnight_live_score"]
        h_score = r.get("heavy_score", 0)
        a_score = r.get("agent_score", 0)
        tier = r.get("heavy_tier", "")
        veto = r.get("heavy_veto", False) or r.get("agent_veto", False)
        risk = r.get("agent_risk_level", "")
        h_reason = r.get("heavy_reason", "")
        a_reason = r.get("agent_reason", "")

        print(f"║ {'─'*78} ║")
        print(f"║  #{i} {r['ts_code']} {str(r.get('name',''))}  |  {str(r.get('industry',''))}  ║")
        print(f"║     最终分={f_score:.4f}  量化分={l_score:.4f}  重审={h_score:.2f}  轻审={a_score:.2f}  ║")
        print(f"║     tier={tier:<6} veto={veto!s:<5} risk={risk:<6}  ║")
        print(f"║     重审: {h_reason}  ║")
        print(f"║     轻审: {a_reason}  ║")
    print(f"╚{'═'*80}╝")

    top5.to_csv(FUSION_DIR / "final_top5.csv", index=False)
    fused.to_csv(FUSION_DIR / "final_fused_all.csv", index=False)
    print(f"\nSaved: {FUSION_DIR}/final_top5.csv")

    # Summary
    print(f"\n总结:")
    print(f"  Top1: {top5.iloc[0]['ts_code']} {top5.iloc[0].get('name','')} final={top5.iloc[0]['final_live_score']:.4f}")
    print(f"  Top2: {top5.iloc[1]['ts_code']} {top5.iloc[1].get('name','')} final={top5.iloc[1]['final_live_score']:.4f}")
    print(f"  Top3: {top5.iloc[2]['ts_code']} {top5.iloc[2].get('name','')} final={top5.iloc[2]['final_live_score']:.4f}")
    print(f"  Top4: {top5.iloc[3]['ts_code']} {top5.iloc[3].get('name','')} final={top5.iloc[3]['final_live_score']:.4f}")
    print(f"  Top5: {top5.iloc[4]['ts_code']} {top5.iloc[4].get('name','')} final={top5.iloc[4]['final_live_score']:.4f}")


if __name__ == "__main__":
    main()
