from __future__ import annotations

"""Pre-close TradingAgents review helpers for live overnight candidates.

The live 14:55 workflow has two clocks:

- prefilter clock, e.g. 14:20-14:40: build Top10/Top15/Top20 buffer;
- final clock, e.g. 14:55: refresh quotes and fuse deterministic live score
  with the agent review produced during the buffer window.

This module keeps the review contract explicit and machine-readable.  The graph
is asked to output a JSON block with per-symbol review scores; if parsing fails,
callers can still keep the raw report for audit and decide whether to fall back
to deterministic ranking.
"""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from dataflows.overnight_news_social_context import build_news_social_context_block, NewsContextResult, summarize_news_social_context
from dataflows.openclaw_context_provider import build_openclaw_prompt_block


REVIEW_REQUIRED_COLUMNS = [
    "ts_code",
    "agent_score",
    "agent_risk_level",
    "agent_veto",
    "agent_adjustment",
    "agent_reason",
]


@dataclass
class AgentReviewParseResult:
    scores: pd.DataFrame
    parsed: bool
    error: str = ""


def _read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p}")
    return pd.read_csv(p)


def load_live_candidate_pool(path: str | Path) -> pd.DataFrame:
    df = _read_csv(path)
    if "ts_code" not in df.columns:
        raise ValueError(f"Live candidate CSV missing ts_code: {path}")
    return df.copy()


def _compact_candidate_records(df: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    keep = [
        c for c in [
            "rank_in_live_day",
            "ts_code",
            "name_x",
            "name_y",
            "industry",
            "market",
            "overnight_live_score",
            "last_price",
            "pre_close",
            "open",
            "high",
            "low",
            "volume",
            "amount",
            "quote_time",
            "history_trade_date",
            "hist_ret_close_1d",
            "hist_ret_close_3d",
            "hist_ret_close_5d",
            "hist_overnight_prev_1d",
            "hist_overnight_prev_3d_mean",
            "hist_overnight_positive_rate_5d",
            "live_return_vs_pre_close",
            "live_return_vs_prev_close",
            "live_range_pos",
            "from_day_high",
            "live_reject_reasons",
            "live_pass_risk_filter",
        ] if c in df.columns
    ]
    records = df[keep].head(limit).copy()
    return json.loads(records.to_json(orient="records", force_ascii=False))


def summarize_live_candidates(df: pd.DataFrame, top_k: int) -> dict[str, Any]:
    out: dict[str, Any] = {
        "candidate_count": int(min(len(df), top_k)),
        "source": "live_preclose_buffer",
    }
    if "industry" in df.columns:
        out["industry_counts"] = df.head(top_k)["industry"].fillna("UNKNOWN").value_counts().to_dict()
    if "market" in df.columns:
        out["market_counts"] = df.head(top_k)["market"].fillna("UNKNOWN").value_counts().to_dict()
    for col in ["overnight_live_score", "live_return_vs_pre_close", "from_day_high", "live_range_pos"]:
        if col in df.columns:
            s = pd.to_numeric(df.head(top_k)[col], errors="coerce")
            out[col] = {
                "min": None if s.dropna().empty else float(s.min()),
                "median": None if s.dropna().empty else float(s.median()),
                "max": None if s.dropna().empty else float(s.max()),
            }
    return out


def build_live_review_payload(
    candidate_pool: pd.DataFrame,
    trade_date: str,
    top_k: int = 15,
    target_top_n: int = 5,
    snapshot_time_hint: str | None = None,
    news_social_context: NewsContextResult | None = None,
    openclaw_context: dict[str, Any] | None = None,
) -> str:
    """Build the prompt/context injected into TradingAgentsGraph."""
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
你是 TradingAgents2.0 的 A 股一夜持股法盘前/尾盘缓冲区审查器。

目标：现在不是最终出票，而是在 14:55 前对候选池做综合分析。请基于候选池，给出每只股票的 agent_score、风险等级、是否 veto，以及排序调整建议。14:55 最终出票时会再刷新实时行情，并融合你的审查结果。

交易规则：
- 买入窗口：收盘前，目标 14:55 后尽快给最终 Top{target_top_n}
- 卖出规则：次日开盘卖出，exit_rule=next_open_sell
- 禁止使用未来信息；只能使用当前候选池和其中可见字段
- 你可以因为新闻/基本面/行业集中/尾盘结构/异常波动/流动性风险而 veto 候选

运行信息：
- trade_date: {trade_date}
- review_buffer_top_k: {top_k}
- target_final_top_n: {target_top_n}
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

请输出两部分：

1. 人类可读分析：按市场、行业、个股、风险、组合构造分别说明。
2. 机器可读 JSON，必须放在如下标记之间：

AGENT_REVIEW_JSON_START
{{
  "trade_date": "{trade_date}",
  "target_top_n": {target_top_n},
  "reviews": [
    {{
      "ts_code": "示例 000001.SZ",
      "agent_score": 0.0到1.0之间，越高越推荐,
      "agent_risk_level": "low|medium|high",
      "agent_veto": false,
      "agent_adjustment": -0.20到0.20之间，正数上调排序，负数下调排序,
      "agent_reason": "一句话说明"
    }}
  ]
}}
AGENT_REVIEW_JSON_END

要求：
- 每个输入候选都必须有一条 reviews 记录。
- 如果无法确认，给 medium 风险，不要编造不存在的事实。
- veto=true 的股票最终 14:55 融合时不能进入 Top3/Top5。
""".strip()


def extract_agent_review_json(text: str) -> dict[str, Any]:
    m = re.search(r"AGENT_REVIEW_JSON_START\s*(\{.*?\})\s*AGENT_REVIEW_JSON_END", text, re.S)
    if not m:
        raise ValueError("AGENT_REVIEW_JSON block not found")
    return json.loads(m.group(1))


def neutral_review_scores(candidate_pool: pd.DataFrame, reason: str = "neutral_fallback") -> pd.DataFrame:
    rows = []
    for _, row in candidate_pool.iterrows():
        rows.append(
            {
                "ts_code": row.get("ts_code"),
                "agent_score": 0.5,
                "agent_risk_level": "medium",
                "agent_veto": False,
                "agent_adjustment": 0.0,
                "agent_reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=REVIEW_REQUIRED_COLUMNS)


def parse_agent_review_scores(text: str, candidate_pool: pd.DataFrame) -> AgentReviewParseResult:
    try:
        data = extract_agent_review_json(text)
        reviews = pd.DataFrame(data.get("reviews", []))
        if reviews.empty or "ts_code" not in reviews.columns:
            raise ValueError("reviews missing or empty")
        for col in REVIEW_REQUIRED_COLUMNS:
            if col not in reviews.columns:
                if col == "agent_veto":
                    reviews[col] = False
                elif col in {"agent_score", "agent_adjustment"}:
                    reviews[col] = 0.0 if col == "agent_adjustment" else 0.5
                else:
                    reviews[col] = ""
        reviews = reviews[REVIEW_REQUIRED_COLUMNS].copy()
        reviews["agent_score"] = pd.to_numeric(reviews["agent_score"], errors="coerce").clip(0, 1).fillna(0.5)
        reviews["agent_adjustment"] = pd.to_numeric(reviews["agent_adjustment"], errors="coerce").clip(-0.2, 0.2).fillna(0.0)
        reviews["agent_veto"] = reviews["agent_veto"].astype(str).str.lower().isin(["true", "1", "yes"])
        expected = pd.DataFrame({"ts_code": candidate_pool["ts_code"].astype(str).unique()})
        merged = expected.merge(reviews, on="ts_code", how="left")
        missing = merged["agent_score"].isna()
        merged.loc[missing, "agent_score"] = 0.5
        merged.loc[missing, "agent_risk_level"] = "medium"
        merged.loc[missing, "agent_veto"] = False
        merged.loc[missing, "agent_adjustment"] = 0.0
        merged.loc[missing, "agent_reason"] = "missing_from_agent_json_neutral"
        return AgentReviewParseResult(merged[REVIEW_REQUIRED_COLUMNS], parsed=True)
    except Exception as exc:
        return AgentReviewParseResult(neutral_review_scores(candidate_pool, reason=f"parse_failed: {exc}"), parsed=False, error=str(exc))


def write_review_artifacts(
    out_dir: str | Path,
    final_state: dict[str, Any],
    candidate_pool: pd.DataFrame,
    decision_text: str,
) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parse_result = parse_agent_review_scores(decision_text, candidate_pool)
    paths = {
        "agent_review_report": out / "agent_review_report.md",
        "agent_review_scores": out / "agent_review_scores.csv",
        "agent_review_state": out / "agent_review_state.json",
        "agent_review_parse_manifest": out / "agent_review_parse_manifest.json",
    }
    paths["agent_review_report"].write_text(decision_text + "\n", encoding="utf-8")
    parse_result.scores.to_csv(paths["agent_review_scores"], index=False)
    paths["agent_review_state"].write_text(json.dumps(final_state, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    paths["agent_review_parse_manifest"].write_text(
        json.dumps({"parsed": parse_result.parsed, "error": parse_result.error}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {k: str(v) for k, v in paths.items()}
