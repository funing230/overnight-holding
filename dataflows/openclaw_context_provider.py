from __future__ import annotations

"""Helpers for loading externally generated OpenClaw overnight context artifacts.

Current scope:
- discover a context directory
- load manifest safely
- read manifest-declared JSON payload files
- build prompt/context blocks for heavy/light review
- export machine-readable feature frames for final fusion
- summarize status for multistage manifests / audits
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class OpenClawContextLoadResult:
    loaded: bool
    context_dir: str | None
    manifest_path: str | None
    manifest: dict[str, Any] | None
    summary: dict[str, Any]
    payload: dict[str, Any] | None = None
    prompt_block: str = ""
    error: str = ""


def _safe_read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_declared_files(ctx_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    files = manifest.get("files") or {}
    if not isinstance(files, dict):
        return out
    for key, rel_path in files.items():
        if not rel_path:
            continue
        file_path = ctx_dir / str(rel_path)
        if not file_path.exists() or not file_path.is_file():
            continue
        try:
            out[key] = _safe_read_json(file_path)
        except Exception:
            continue
    return out


def _top_list(items: Any, limit: int = 5) -> list[Any]:
    if isinstance(items, list):
        return items[:limit]
    return []


def _summarize_payload_data(manifest: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    ticker_event_map = payload.get("ticker_event_map") or []
    macro_news = payload.get("macro_news") or []
    social_hotspots = payload.get("social_hotspots") or []
    theme_context = payload.get("theme_context") or []
    risk_flags = payload.get("risk_flags") or []
    return {
        **summarize_openclaw_context_manifest(manifest),
        "payload_keys": sorted(payload.keys()),
        "ticker_event_count": len(ticker_event_map) if isinstance(ticker_event_map, list) else 0,
        "macro_news_count": len(macro_news) if isinstance(macro_news, list) else 0,
        "social_hotspots_count": len(social_hotspots) if isinstance(social_hotspots, list) else 0,
        "theme_context_count": len(theme_context) if isinstance(theme_context, list) else 0,
        "risk_flags_count": len(risk_flags) if isinstance(risk_flags, list) else 0,
    }


def build_openclaw_prompt_block(payload: dict[str, Any] | None, summary: dict[str, Any] | None = None) -> str:
    if not payload:
        return ""
    summary = summary or {}
    macro_news = _top_list(payload.get("macro_news"), limit=3)
    social_hotspots = _top_list(payload.get("social_hotspots"), limit=3)
    ticker_event_map = _top_list(payload.get("ticker_event_map"), limit=5)
    risk_flags = _top_list(payload.get("risk_flags"), limit=5)
    compact = {
        "summary": summary,
        "macro_news_top": macro_news,
        "social_hotspots_top": social_hotspots,
        "ticker_event_top": ticker_event_map,
        "risk_flags_top": risk_flags,
    }
    return (
        "\n\n附加上下文：以下是 OpenClaw 外部抓取增强信息（仅作 soft signal / risk context，不可替代原始行情与候选池字段）。\n"
        "使用规则：\n"
        "- 只能把它当作催化、风险、主题、舆情辅助线索。\n"
        "- 如与候选池原始字段冲突，优先保守，不要编造结论。\n"
        "- 若摘要证据不足，宁可降低权重，不要强行 veto。\n"
        "```json\n"
        + json.dumps(compact, ensure_ascii=False, indent=2)
        + "\n```\n"
    )


def build_openclaw_feature_frame(payload: dict[str, Any] | None) -> pd.DataFrame:
    columns = [
        "ts_code",
        "openclaw_event_strength",
        "openclaw_positive_signal",
        "openclaw_risk_penalty",
        "openclaw_theme_count",
        "openclaw_theme_names",
        "openclaw_macro_mentions",
        "openclaw_social_mentions",
        "openclaw_feature_score",
        "openclaw_risk_flags",
        "openclaw_catalyst_summary",
    ]
    if not payload:
        return pd.DataFrame(columns=columns)

    ticker_events = payload.get("ticker_event_map") or []
    social_hotspots = payload.get("social_hotspots") or []
    macro_news = payload.get("macro_news") or []
    risk_flags = payload.get("risk_flags") or []

    social_mentions: dict[str, int] = {}
    for item in social_hotspots if isinstance(social_hotspots, list) else []:
        for code in item.get("matched_tickers") or []:
            key = str(code).strip().upper()
            if key:
                social_mentions[key] = social_mentions.get(key, 0) + 1

    macro_theme_hits: dict[str, int] = {}
    for item in macro_news if isinstance(macro_news, list) else []:
        tags = item.get("matched_tickers") or item.get("related_tickers") or []
        for code in tags:
            key = str(code).strip().upper()
            if key:
                macro_theme_hits[key] = macro_theme_hits.get(key, 0) + 1

    risk_map: dict[str, list[str]] = {}
    for item in risk_flags if isinstance(risk_flags, list) else []:
        key = str(item.get("ts_code", "")).strip().upper()
        if not key:
            continue
        label = str(item.get("risk") or item.get("label") or item.get("severity") or "risk").strip()
        if label:
            risk_map.setdefault(key, []).append(label)

    rows: list[dict[str, Any]] = []
    for item in ticker_events if isinstance(ticker_events, list) else []:
        key = str(item.get("ts_code", "")).strip().upper()
        if not key:
            continue
        event_strength = float(item.get("event_strength") or 0.0)
        sentiment = str(item.get("sentiment") or "").strip().lower()
        risk_level = str(item.get("risk_level") or "").strip().lower()
        theme_tags = [str(x).strip() for x in (item.get("theme_tags") or []) if str(x).strip()]
        positive_signal = 0.03 if sentiment in {"positive", "bullish", "risk_on"} else (0.0 if sentiment not in {"negative", "bearish", "risk_off"} else -0.03)
        risk_penalty = {"low": 0.0, "medium": 0.02, "high": 0.05}.get(risk_level, 0.01 if risk_level else 0.0)
        feature_score = max(min(0.08 * event_strength + positive_signal - risk_penalty, 0.08), -0.08)
        rows.append(
            {
                "ts_code": key,
                "openclaw_event_strength": event_strength,
                "openclaw_positive_signal": positive_signal,
                "openclaw_risk_penalty": risk_penalty,
                "openclaw_theme_count": len(theme_tags),
                "openclaw_theme_names": "|".join(theme_tags),
                "openclaw_macro_mentions": int(macro_theme_hits.get(key, 0)),
                "openclaw_social_mentions": int(social_mentions.get(key, 0)),
                "openclaw_feature_score": feature_score,
                "openclaw_risk_flags": "|".join(risk_map.get(key, [])),
                "openclaw_catalyst_summary": str(item.get("catalyst_summary") or "").strip(),
            }
        )

    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for col in [
        "openclaw_event_strength",
        "openclaw_positive_signal",
        "openclaw_risk_penalty",
        "openclaw_theme_count",
        "openclaw_macro_mentions",
        "openclaw_social_mentions",
        "openclaw_feature_score",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values(["openclaw_feature_score", "openclaw_event_strength", "ts_code"], ascending=[False, False, True]).drop_duplicates(subset=["ts_code"], keep="first")
    return df[columns].reset_index(drop=True)


def find_latest_openclaw_context(trade_date: str, root: str | Path) -> Path | None:
    base = Path(root) / str(trade_date)
    if not base.exists() or not base.is_dir():
        return None
    candidates = [p for p in base.iterdir() if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def summarize_openclaw_context_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files") or {}
    sources = manifest.get("sources") or {}
    degrade_reasons = manifest.get("degrade_reasons") or []
    return {
        "trade_date": manifest.get("trade_date"),
        "run_ts": manifest.get("run_ts"),
        "generated_at": manifest.get("generated_at"),
        "candidate_source": manifest.get("candidate_source"),
        "candidate_count": manifest.get("candidate_count"),
        "degraded": bool(manifest.get("degraded", False)),
        "degrade_reasons": list(degrade_reasons) if isinstance(degrade_reasons, list) else [str(degrade_reasons)],
        "source_flags": sources,
        "file_count": len(files) if isinstance(files, dict) else 0,
        "file_keys": sorted(files.keys()) if isinstance(files, dict) else [],
    }


def load_openclaw_context(context_dir: str | Path) -> OpenClawContextLoadResult:
    ctx_dir = Path(context_dir)
    manifest_path = ctx_dir / "manifest.json"
    if not ctx_dir.exists() or not ctx_dir.is_dir():
        return OpenClawContextLoadResult(
            loaded=False,
            context_dir=str(ctx_dir),
            manifest_path=str(manifest_path),
            manifest=None,
            payload=None,
            prompt_block="",
            summary={"loaded": False, "reason": "context_dir_not_found"},
            error="context_dir_not_found",
        )
    if not manifest_path.exists():
        return OpenClawContextLoadResult(
            loaded=False,
            context_dir=str(ctx_dir),
            manifest_path=str(manifest_path),
            manifest=None,
            payload=None,
            prompt_block="",
            summary={"loaded": False, "reason": "manifest_not_found"},
            error="manifest_not_found",
        )
    try:
        manifest = _safe_read_json(manifest_path)
    except Exception as exc:
        return OpenClawContextLoadResult(
            loaded=False,
            context_dir=str(ctx_dir),
            manifest_path=str(manifest_path),
            manifest=None,
            payload=None,
            prompt_block="",
            summary={"loaded": False, "reason": "manifest_parse_failed"},
            error=f"manifest_parse_failed:{type(exc).__name__}:{exc}",
        )

    payload = _load_declared_files(ctx_dir, manifest)
    summary = _summarize_payload_data(manifest, payload)
    summary["loaded"] = True
    summary["context_dir"] = str(ctx_dir)
    prompt_block = build_openclaw_prompt_block(payload, summary=summary)
    return OpenClawContextLoadResult(
        loaded=True,
        context_dir=str(ctx_dir),
        manifest_path=str(manifest_path),
        manifest=manifest,
        payload=payload,
        prompt_block=prompt_block,
        summary=summary,
        error="",
    )


def resolve_openclaw_context(
    trade_date: str,
    context_dir: str | Path | None = None,
    root: str | Path | None = None,
) -> OpenClawContextLoadResult:
    if context_dir:
        return load_openclaw_context(context_dir)
    if root:
        latest = find_latest_openclaw_context(trade_date, root)
        if latest is not None:
            return load_openclaw_context(latest)
        return OpenClawContextLoadResult(
            loaded=False,
            context_dir=None,
            manifest_path=None,
            manifest=None,
            payload=None,
            prompt_block="",
            summary={"loaded": False, "reason": "no_context_found_under_root", "trade_date": trade_date, "root": str(root)},
            error="no_context_found_under_root",
        )
    return OpenClawContextLoadResult(
        loaded=False,
        context_dir=None,
        manifest_path=None,
        manifest=None,
        payload=None,
        prompt_block="",
        summary={"loaded": False, "reason": "openclaw_context_not_requested"},
        error="openclaw_context_not_requested",
    )
