from __future__ import annotations

"""Heavy TradingAgents review helpers for Top50 -> Top15 overnight live funnel.

Heavy review runs before the 14:30 light review.  It should be earlier and more
research-oriented than light review, but still batch-oriented: one Top50 pool in,
ranked Top15 plus machine-readable scores out.  It does not use future labels.
"""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from dataflows.overnight_news_social_context import build_news_social_context_block, NewsContextResult, summarize_news_social_context
from dataflows.openclaw_context_provider import build_openclaw_prompt_block

from dataflows.overnight_live_review_provider import (
    load_live_candidate_pool,
    summarize_live_candidates,
    _compact_candidate_records,
)


HEAVY_REVIEW_REQUIRED_COLUMNS = [
    "ts_code",
    "heavy_score",
    "heavy_tier",
    "heavy_veto",
    "heavy_adjustment",
    "heavy_keep_rank",
    "heavy_reason",
    "heavy_risk_flags",
]


@dataclass
class HeavyReviewParseResult:
    scores: pd.DataFrame
    parsed: bool
    error: str = ""


def build_heavy_review_payload(
    candidate_pool: pd.DataFrame,
    trade_date: str,
    top_k: int = 50,
    target_top_n: int = 15,
    snapshot_time_hint: str | None = None,
    news_social_context: NewsContextResult | None = None,
    openclaw_context: dict[str, Any] | None = None,
) -> str:
    """Build heavy review prompt for Top50 -> Top15 research compression."""
    records = _compact_candidate_records(candidate_pool, limit=top_k)
    summary = summarize_live_candidates(candidate_pool, top_k=top_k)
    if news_social_context is not None:
        summary["news_social_context"] = summarize_news_social_context(news_social_context)
    if openclaw_context is not None:
        summary["openclaw_context"] = openclaw_context.get("summary")
    snapshot_line = f"- snapshot_time_hint: {snapshot_time_hint}\n" if snapshot_time_hint else ""
    news_social_block = build_news_social_context_block(news_social_context) if news_social_context is not None else ""
    openclaw_block = build_openclaw_prompt_block(openclaw_context.get("payload"), openclaw_context.get("summary")) if openclaw_context is not None else ""
    return f"""
你是 TradingAgents2.0 的 A 股一夜持股法重度候选池研究器。

目标：现在不是最终 14:55 出票，也不是轻量尾盘复核；你的任务是在更早时间把 deterministic Top{top_k} 候选池压缩成高质量 Top{target_top_n} 研究池，供后续 14:30 light review 和 14:55 最终融合使用。

交易规则：
- 买入窗口：收盘前，目标最终 14:55 后输出 Top3/Top5
- 卖出规则：次日开盘卖出，exit_rule=next_open_sell
- 禁止使用未来信息；只能使用当前候选池和其中可见字段
- 不要编造新闻、公告、财报或未提供事实
- 你可以基于行业集中、尾盘结构、波动异常、流动性、短期趋势质量、历史 overnight 稳定性、组合构造给出保留/降级/veto

运行信息：
- trade_date: {trade_date}
- heavy_review_top_k: {top_k}
- target_research_top_n: {target_top_n}
{snapshot_line}
候选池摘要：
```json
{json.dumps(summary, ensure_ascii=False, indent=2)}
```

候选池明细：
```json
{json.dumps(records, ensure_ascii=False, indent=2)}
```
{news_social_block}
{openclaw_block}

请严格按以下顺序输出：

第一部分：机器可读 JSON，必须最先输出，必须放在如下标记之间：

HEAVY_REVIEW_JSON_START
{{
  "trade_date": "{trade_date}",
  "target_top_n": {target_top_n},
  "top_picks": [
    {{
      "ts_code": "示例 000001.SZ",
      "heavy_score": 0.0到1.0之间，越高越推荐进入研究池,
      "heavy_tier": "core|watch",
      "heavy_veto": false,
      "heavy_adjustment": -0.20到0.20之间，正数上调，负数下调,
      "heavy_keep_rank": 1,
      "heavy_reason": "一句话说明，<=20字",
      "heavy_risk_flags": ["短标签1", "短标签2，最多2个"]
    }}
  ],
  "rejects": [
    {{
      "ts_code": "示例 000002.SZ",
      "heavy_score": 0.0到0.5之间,
      "heavy_tier": "reject",
      "heavy_veto": true,
      "heavy_adjustment": -0.20到0.00之间,
      "heavy_keep_rank": 999,
      "heavy_reason": "拒绝理由，<=20字",
      "heavy_risk_flags": ["短标签1", "短标签2，最多2个"]
    }}
  ],
  "summary": {{
    "core_count": 0,
    "watch_top15_count": 0,
    "reject_count": 0,
    "notes": "不超过120字；未出现在 top_picks/rejects 的默认视为 watch"
  }}
}}
HEAVY_REVIEW_JSON_END

第二部分：人类可读分析，最多 120 中文字；如果 JSON 已较长，可不写或只写 1-2 句极简总结。

要求：
- 不要输出长篇推理过程。
- 不要输出 <thinking>、Chain-of-Thought、草稿、逐条展开的 50 只点评。
- 不要在 JSON 之前输出任何文字。
- top_picks 最多保留 {target_top_n} 条，按 heavy_keep_rank 从 1 开始排序。
- rejects 只写明确不应进入研究池/应 veto 的股票。
- 未出现在 top_picks 或 rejects 的股票，程序会自动视为 watch；不要重复枚举。
- heavy_tier=reject 或 heavy_veto=true 的股票原则上不进入 Top{target_top_n}。
- 如果无法确认，宁可不写入 top_picks/rejects，也不要编造事实。
- 为节省输出长度，优先保持 JSON 完整，summary 宁可极短。
""".strip()


def extract_heavy_review_json(text: str) -> dict[str, Any]:
    m = re.search(r"HEAVY_REVIEW_JSON_START\s*(\{.*?\})\s*HEAVY_REVIEW_JSON_END", text, re.S)
    if m:
        return json.loads(m.group(1))

    start = re.search(r"HEAVY_REVIEW_JSON_START\s*", text, re.S)
    if not start:
        raise ValueError("HEAVY_REVIEW_JSON block not found")

    tail = text[start.end():]
    first_brace = tail.find("{")
    if first_brace < 0:
        raise ValueError("HEAVY_REVIEW_JSON start found but JSON body missing")

    candidate = tail[first_brace:]
    decoder = json.JSONDecoder()
    try:
        obj, _end = decoder.raw_decode(candidate)
        return obj
    except Exception:
        repaired = _extract_first_balanced_json_object(candidate)
        if repaired is None:
            raise ValueError("HEAVY_REVIEW_JSON start found but JSON appears truncated")
        return json.loads(repaired)


def _extract_first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def neutral_heavy_review_scores(candidate_pool: pd.DataFrame, reason: str = "neutral_heavy_fallback") -> pd.DataFrame:
    rows = []
    for i, row in candidate_pool.reset_index(drop=True).iterrows():
        rows.append(
            {
                "ts_code": row.get("ts_code"),
                "heavy_score": 0.5,
                "heavy_tier": "watch",
                "heavy_veto": False,
                "heavy_adjustment": 0.0,
                "heavy_keep_rank": i + 1,
                "heavy_reason": reason,
                "heavy_risk_flags": "[]",
            }
        )
    return pd.DataFrame(rows, columns=HEAVY_REVIEW_REQUIRED_COLUMNS)


def _fallback_heavy_scores_from_ranked_text(text: str, candidate_pool: pd.DataFrame) -> pd.DataFrame | None:
    """Best-effort fallback when a reasoning model omits/truncates JSON.

    Prefer the last compact ranked list in the report.  Reasoning models often
    first enumerate all candidates, then later provide a final Top15; using the
    first occurrence would accidentally recover the evaluation order instead of
    the final decision.

    Accept both full ts_code forms (e.g. ``000333.SZ``) and bare 6-digit A-share
    codes (e.g. ``000333``), since truncated reasoning outputs often drop the
    exchange suffix in their final ranked list.
    """
    candidate_codes = candidate_pool["ts_code"].astype(str)
    candidates = set(candidate_codes)
    bare_to_full: dict[str, str] = {}
    for code in candidates:
        bare = code.split(".", 1)[0]
        bare_to_full.setdefault(bare, code)

    line_re = re.compile(r"^\s*\d{1,2}[\.)、]\s+([0-9]{6}(?:\.(?:SZ|SH))?)\b")
    groups: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        m = line_re.search(line)
        if m:
            raw_code = m.group(1)
            normalized = raw_code if raw_code in candidates else bare_to_full.get(raw_code)
            if normalized is not None:
                current.append(normalized)
                continue
        if current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)

    if not groups:
        inline_codes: list[str] = []
        for raw_code in re.findall(r"([0-9]{6}(?:\.(?:SZ|SH))?)", text):
            normalized = raw_code if raw_code in candidates else bare_to_full.get(raw_code)
            if normalized is not None:
                inline_codes.append(normalized)
        dedup_inline: list[str] = []
        seen_inline: set[str] = set()
        for code in inline_codes:
            if code not in seen_inline:
                seen_inline.add(code)
                dedup_inline.append(code)
        if 5 <= len(dedup_inline) <= 25:
            groups = [dedup_inline]

    if not groups:
        return None
    plausible = [g for g in groups if 5 <= len(g) <= 25]
    chosen = plausible[-1] if plausible else groups[-1]

    matches: list[str] = []
    seen: set[str] = set()
    for code in chosen:
        if code not in seen:
            seen.add(code)
            matches.append(code)
    if not matches:
        return None

    rows = []
    rank_map = {code: i + 1 for i, code in enumerate(matches)}
    for i, row in candidate_pool.reset_index(drop=True).iterrows():
        code = str(row.get("ts_code"))
        bare_code = code.split(".", 1)[0]
        if code in rank_map:
            keep_rank = rank_map[code]
            heavy_score = max(0.55, 0.92 - 0.015 * (keep_rank - 1))
            tier = "core" if keep_rank <= 8 else "watch"
            adjustment = max(0.0, 0.08 - 0.004 * (keep_rank - 1))
            reason = "fallback_ranked_text_top15"
        else:
            rejected = re.search(
                rf"(?:reject|veto|剔除|硬性排除).*(?:{re.escape(code)}|{re.escape(bare_code)})|(?:{re.escape(code)}|{re.escape(bare_code)}).*(?:reject|veto|剔除|硬性排除)",
                text,
                re.I,
            ) is not None
            keep_rank = len(candidate_pool) + i + 1
            heavy_score = 0.25 if rejected else 0.5
            tier = "reject" if rejected else "watch"
            adjustment = -0.08 if rejected else 0.0
            reason = "fallback_ranked_text_reject" if rejected else "fallback_ranked_text_neutral"
        rows.append(
            {
                "ts_code": code,
                "heavy_score": heavy_score,
                "heavy_tier": tier,
                "heavy_veto": tier == "reject",
                "heavy_adjustment": adjustment,
                "heavy_keep_rank": keep_rank,
                "heavy_reason": reason,
                "heavy_risk_flags": "[]",
            }
        )
    return pd.DataFrame(rows, columns=HEAVY_REVIEW_REQUIRED_COLUMNS)


def _normalize_heavy_review_records(records: pd.DataFrame, default_start_rank: int = 1) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(columns=HEAVY_REVIEW_REQUIRED_COLUMNS)
    reviews = records.copy()
    for col in HEAVY_REVIEW_REQUIRED_COLUMNS:
        if col not in reviews.columns:
            if col == "heavy_veto":
                reviews[col] = False
            elif col == "heavy_score":
                reviews[col] = 0.5
            elif col == "heavy_adjustment":
                reviews[col] = 0.0
            elif col == "heavy_keep_rank":
                reviews[col] = range(default_start_rank, default_start_rank + len(reviews))
            elif col == "heavy_tier":
                reviews[col] = "watch"
            elif col == "heavy_risk_flags":
                reviews[col] = "[]"
            else:
                reviews[col] = ""
    reviews = reviews[HEAVY_REVIEW_REQUIRED_COLUMNS].copy()
    reviews["ts_code"] = reviews["ts_code"].astype(str)
    reviews["heavy_score"] = pd.to_numeric(reviews["heavy_score"], errors="coerce").clip(0, 1).fillna(0.5)
    reviews["heavy_adjustment"] = pd.to_numeric(reviews["heavy_adjustment"], errors="coerce").clip(-0.2, 0.2).fillna(0.0)
    reviews["heavy_keep_rank"] = pd.to_numeric(reviews["heavy_keep_rank"], errors="coerce")
    reviews["heavy_veto"] = reviews["heavy_veto"].astype(str).str.lower().isin(["true", "1", "yes"])
    reviews["heavy_tier"] = reviews["heavy_tier"].astype(str).str.lower().where(
        reviews["heavy_tier"].astype(str).str.lower().isin(["core", "watch", "reject"]),
        "watch",
    )
    reviews["heavy_risk_flags"] = reviews["heavy_risk_flags"].apply(
        lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, list) else str(x)
    )
    reviews = reviews.drop_duplicates(subset=["ts_code"], keep="first")
    return reviews


def _materialize_schema_scores(data: dict[str, Any], candidate_pool: pd.DataFrame, target_top_n: int | None = None) -> pd.DataFrame:
    expected = pd.DataFrame({"ts_code": candidate_pool["ts_code"].astype(str).unique()})

    if isinstance(data.get("reviews"), list):
        reviews = _normalize_heavy_review_records(pd.DataFrame(data.get("reviews", [])))
        if reviews.empty or "ts_code" not in reviews.columns:
            raise ValueError("reviews missing or empty")
        merged = expected.merge(reviews, on="ts_code", how="left")
        missing = merged["heavy_score"].isna()
        merged.loc[missing, "heavy_score"] = 0.5
        merged.loc[missing, "heavy_tier"] = "watch"
        merged.loc[missing, "heavy_veto"] = False
        merged.loc[missing, "heavy_adjustment"] = 0.0
        merged.loc[missing, "heavy_reason"] = "missing_from_heavy_json_neutral"
        merged.loc[missing, "heavy_risk_flags"] = "[]"
        fallback_rank = pd.Series(range(1, len(merged) + 1), index=merged.index)
        merged["heavy_keep_rank"] = pd.to_numeric(merged["heavy_keep_rank"], errors="coerce").fillna(fallback_rank)
        return merged[HEAVY_REVIEW_REQUIRED_COLUMNS]

    top_picks = _normalize_heavy_review_records(pd.DataFrame(data.get("top_picks", [])), default_start_rank=1)
    rejects = _normalize_heavy_review_records(pd.DataFrame(data.get("rejects", [])), default_start_rank=999)

    if top_picks.empty and rejects.empty:
        raise ValueError("top_picks/rejects missing or empty")

    if not top_picks.empty:
        top_picks["heavy_tier"] = top_picks["heavy_tier"].where(top_picks["heavy_tier"].isin(["core", "watch"]), "watch")
        top_picks["heavy_veto"] = False
    if not rejects.empty:
        rejects["heavy_tier"] = "reject"
        rejects["heavy_veto"] = True
        rejects["heavy_keep_rank"] = pd.to_numeric(rejects["heavy_keep_rank"], errors="coerce").fillna(999)

    explicit = pd.concat([top_picks, rejects], ignore_index=True) if (not top_picks.empty or not rejects.empty) else pd.DataFrame(columns=HEAVY_REVIEW_REQUIRED_COLUMNS)
    explicit = explicit.drop_duplicates(subset=["ts_code"], keep="first")

    merged = expected.merge(explicit, on="ts_code", how="left")
    missing = merged["heavy_score"].isna()
    target_top_n = int(target_top_n or data.get("target_top_n") or 15)
    default_start = max(100, target_top_n + 1)
    default_ranks = pd.Series(range(default_start, default_start + int(missing.sum())), index=merged.index[missing])
    merged.loc[missing, "heavy_score"] = 0.5
    merged.loc[missing, "heavy_tier"] = "watch"
    merged.loc[missing, "heavy_veto"] = False
    merged.loc[missing, "heavy_adjustment"] = 0.0
    merged.loc[missing, "heavy_reason"] = "default_watch_unmentioned"
    merged.loc[missing, "heavy_risk_flags"] = "[]"
    merged.loc[missing, "heavy_keep_rank"] = default_ranks
    merged["heavy_keep_rank"] = pd.to_numeric(merged["heavy_keep_rank"], errors="coerce")
    return merged[HEAVY_REVIEW_REQUIRED_COLUMNS]


def parse_heavy_review_scores(text: str, candidate_pool: pd.DataFrame) -> HeavyReviewParseResult:
    try:
        data = extract_heavy_review_json(text)
        scores = _materialize_schema_scores(data, candidate_pool, target_top_n=int(data.get("target_top_n") or 15))
        return HeavyReviewParseResult(scores, parsed=True)
    except Exception as exc:
        fallback = _fallback_heavy_scores_from_ranked_text(text, candidate_pool)
        if fallback is not None:
            return HeavyReviewParseResult(fallback, parsed=True, error=f"json_parse_failed_used_ranked_text_fallback: {exc}")
        return HeavyReviewParseResult(neutral_heavy_review_scores(candidate_pool, reason=f"parse_failed: {exc}"), parsed=False, error=str(exc))


def apply_heavy_selection(candidate_pool: pd.DataFrame, scores: pd.DataFrame, target_top_n: int = 15) -> pd.DataFrame:
    out = candidate_pool.copy().merge(scores, on="ts_code", how="left")
    out["heavy_score"] = pd.to_numeric(out.get("heavy_score"), errors="coerce").fillna(0.5)
    out["heavy_adjustment"] = pd.to_numeric(out.get("heavy_adjustment"), errors="coerce").fillna(0.0)
    out["heavy_keep_rank"] = pd.to_numeric(out.get("heavy_keep_rank"), errors="coerce")
    out["heavy_veto"] = out.get("heavy_veto", False).astype(str).str.lower().isin(["true", "1", "yes"])
    out["heavy_tier"] = out.get("heavy_tier", "watch").astype(str).str.lower()
    tier_bonus = out["heavy_tier"].map({"core": 0.10, "watch": 0.0, "reject": -0.20}).fillna(0.0)
    base_live = pd.to_numeric(out.get("overnight_live_score"), errors="coerce").fillna(0.5)
    rank_bonus = -0.001 * out["heavy_keep_rank"].fillna(len(out) + 1)
    out["heavy_selection_score"] = 0.55 * base_live + 0.45 * out["heavy_score"] + out["heavy_adjustment"] + tier_bonus + rank_bonus
    out.loc[out["heavy_veto"] | out["heavy_tier"].eq("reject"), "heavy_selection_score"] = -999.0
    out["rank_in_heavy_review"] = out["heavy_selection_score"].rank(method="first", ascending=False)
    selected = out.sort_values(["rank_in_heavy_review", "ts_code"]).head(target_top_n).copy()
    return selected.reset_index(drop=True)


def write_heavy_review_artifacts(
    out_dir: str | Path,
    final_state: dict[str, Any],
    candidate_pool: pd.DataFrame,
    decision_text: str,
    target_top_n: int = 15,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parse_result = parse_heavy_review_scores(decision_text, candidate_pool)
    selected = apply_heavy_selection(candidate_pool, parse_result.scores, target_top_n=target_top_n)
    paths = {
        "heavy_review_report": out / "heavy_review_report.md",
        "heavy_review_scores": out / "heavy_review_scores.csv",
        "heavy_selected_top15": out / "heavy_selected_top15.csv",
        "heavy_review_state": out / "heavy_review_state.json",
        "heavy_review_parse_manifest": out / "heavy_review_parse_manifest.json",
    }
    paths["heavy_review_report"].write_text(decision_text + "\n", encoding="utf-8")
    parse_result.scores.to_csv(paths["heavy_review_scores"], index=False)
    selected.to_csv(paths["heavy_selected_top15"], index=False)
    paths["heavy_review_state"].write_text(json.dumps(final_state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    paths["heavy_review_parse_manifest"].write_text(
        json.dumps({"parsed": parse_result.parsed, "error": parse_result.error}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {k: str(v) for k, v in paths.items()}
