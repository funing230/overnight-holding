from __future__ import annotations

"""Lightweight news + social-sentiment context builder for overnight live review.

Goal:
- build a structured News Top10 and Social-sentiment Top10 context block
- optionally enrich recall TopK with a minimal social_hot_context layer
- use only currently available repository vendors/tools
- avoid claiming a dedicated social-media feed when only news aggregation exists
"""

import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests

from dataflows.interface import route_to_vendor

# Optional providers — imported lazily to avoid hard deps
_xueqiu_provider = None
_twitter_provider = None


def _get_xueqiu_provider():
    global _xueqiu_provider
    if _xueqiu_provider is None:
        try:
            from dataflows.xueqiu_hot_provider import build_xueqiu_context
            _xueqiu_provider = build_xueqiu_context
        except ImportError:
            _xueqiu_provider = False
    return _xueqiu_provider


def _get_twitter_provider():
    global _twitter_provider
    if _twitter_provider is None:
        try:
            from dataflows.twitter_sentiment_provider import build_twitter_context
            _twitter_provider = build_twitter_context
        except ImportError:
            _twitter_provider = False
    return _twitter_provider


DEFAULT_SOCIAL_HOT_SOURCES = ["weibo", "zhihu"]
TOPHUB_SOURCE_URLS = {
    "weibo": "https://tophub.today/n/KqndgxeLl9",
    "zhihu": "https://tophub.today/n/mproPpoq6O",
}
_SOCIAL_REQUEST_TIMEOUT = 12

SOURCE_LABELS = {
    "weibo": "微博",
    "zhihu": "知乎",
    "toutiao": "今日头条",
    "baidu": "百度热点",
    "bilibili": "B站",
    "36kr": "36氪",
    "ithome": "IT之家",
}

MANUAL_TICKER_ALIASES: dict[str, list[str]] = {
    "000333.SZ": ["美的", "美的集团", "Midea"],
    "600941.SH": ["中国移动", "移动", "China Mobile"],
    "601728.SH": ["中国电信", "电信", "China Telecom"],
    "600519.SH": ["贵州茅台", "茅台", "Kweichow Moutai"],
}

THEME_KEYWORDS = {
    "ai_terminal": ["人工智能", "ai", "智能化", "大模型", "终端", "智能终端", "ai图", "算力"],
    "ai_infra": ["算力", "云", "云网", "通信", "运营商", "国标", "腾讯", "终端智能化"],
    "robotics": ["机器人", "自动驾驶", "智能制造", "装备公司", "装备", "工业自动化"],
    "new_energy": ["新能源", "储能", "锂", "光伏", "精锡", "电池", "逆变器"],
    "consumer_travel": ["离境退税", "入境消费", "消费", "旅游", "世界杯转播"],
    "macro_diplomacy": ["中美", "特朗普访华", "访华", "会谈", "领导人致辞", "普京"],
}

THEME_LABELS = {
    "ai_terminal": "AI/终端智能化",
    "ai_infra": "AI/算力基础设施",
    "robotics": "机器人/智能制造",
    "new_energy": "新能源/储能/锂电",
    "consumer_travel": "消费/离境退税/出行",
    "macro_diplomacy": "宏观外交/风险偏好",
}

THEME_INDUSTRY_HINTS = {
    "ai_terminal": ["半导体", "元器件", "IT设备", "通信设备"],
    "ai_infra": ["电信运营", "通信设备", "IT设备", "半导体"],
    "robotics": ["电气设备", "工程机械", "IT设备"],
    "new_energy": ["电气设备", "汽车整车", "小金属", "化工原料"],
    "consumer_travel": ["旅游服务", "全国地产", "家居用品"],
    "macro_diplomacy": ["电信运营", "通信设备", "半导体", "元器件", "汽车整车"],
}

THEME_TICKER_HINTS = {
    "ai_terminal": ["688256.SH", "000100.SZ", "002236.SZ", "002415.SZ"],
    "ai_infra": ["600050.SH", "601728.SH", "000063.SZ", "688256.SH"],
    "robotics": ["300274.SZ", "000157.SZ", "600031.SH", "601877.SH", "605117.SH"],
    "new_energy": ["300274.SZ", "002459.SZ", "601877.SH", "605117.SH", "002460.SZ", "002594.SZ", "688303.SH", "603260.SH"],
    "consumer_travel": ["601888.SH", "001979.SZ", "603833.SH"],
    "macro_diplomacy": ["688256.SH", "000063.SZ", "000100.SZ", "600050.SH", "601728.SH", "002594.SZ"],
}

THEME_SOURCE_BONUS = {
    "news": 0.03,
    "social": 0.02,
}


BLACKLIST_GENERIC_TERMS = {
    "中国",
    "集团",
    "股份",
    "有限公司",
    "科技",
    "智能",
    "控股",
    "电气",
    "医药",
    "银行",
    "证券",
    "能源",
    "材料",
    "制造",
    "电子",
}


@dataclass
class NewsContextResult:
    news_top10: list[dict[str, Any]]
    social_top10: list[dict[str, Any]]
    global_news_top10: list[dict[str, Any]]
    vendor: str
    notes: list[str]
    social_hot_features: list[dict[str, Any]] = field(default_factory=list)
    social_hot_summary: dict[str, Any] = field(default_factory=dict)
    theme_hot_features: list[dict[str, Any]] = field(default_factory=list)
    theme_hot_summary: dict[str, Any] = field(default_factory=dict)
    xueqiu_hot_features: list[dict[str, Any]] = field(default_factory=list)
    xueqiu_hot_summary: dict[str, Any] = field(default_factory=dict)
    twitter_features: list[dict[str, Any]] = field(default_factory=list)
    twitter_summary: dict[str, Any] = field(default_factory=dict)


def _normalize_ts_code_for_vendor(ts_code: str) -> str:
    code = str(ts_code).strip().upper()
    m = re.match(r"^(\d{6})\.(SZ|SH|BJ)$", code)
    return m.group(1) if m else code


def _safe_json_loads(text: str) -> Any | None:
    try:
        return json.loads(text)
    except Exception:
        return None


def _parse_markdown_news(text: str, ticker: str | None = None) -> list[dict[str, Any]]:
    if not text or text.startswith("No news found") or text.startswith("Error fetching") or text.startswith("No global news"):
        return []

    articles: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("### "):
            if current:
                articles.append(current)
            title_line = line[4:].strip()
            source = ""
            m = re.match(r"^(.*?)\s*\(source:\s*(.*?)\)\s*$", title_line)
            if m:
                title = m.group(1).strip()
                source = m.group(2).strip()
            else:
                title = title_line
            current = {
                "ticker": ticker,
                "title": title,
                "summary": "",
                "source": source,
                "link": "",
                "published_at": "",
                "sentiment": None,
                "relevance": None,
                "channel": "news",
                "raw": [],
            }
            continue
        if current is None:
            continue
        if line.startswith("Published:"):
            current["published_at"] = line.split(":", 1)[1].strip()
        elif line.startswith("Link:"):
            current["link"] = line.split(":", 1)[1].strip()
        elif line:
            current["raw"].append(line)
    if current:
        articles.append(current)

    for item in articles:
        raw = item.pop("raw", [])
        item["summary"] = " ".join(raw).strip()
    return articles


def _parse_alpha_vantage_news(payload: Any, ticker: str | None = None) -> list[dict[str, Any]]:
    if isinstance(payload, str):
        payload = _safe_json_loads(payload)
    if not isinstance(payload, dict):
        return []
    feed = payload.get("feed", []) or []
    out: list[dict[str, Any]] = []
    for article in feed:
        ticker_sentiments = article.get("ticker_sentiment", []) or []
        relevance = None
        sentiment = None
        for row in ticker_sentiments:
            if ticker and str(row.get("ticker", "")).upper() == str(ticker).upper():
                relevance = _to_float(row.get("relevance_score"))
                sentiment = _to_float(row.get("ticker_sentiment_score"))
                break
        if sentiment is None:
            sentiment = _to_float(article.get("overall_sentiment_score"))
        out.append(
            {
                "ticker": ticker,
                "title": article.get("title", ""),
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "link": article.get("url", ""),
                "published_at": article.get("time_published", ""),
                "sentiment": sentiment,
                "relevance": relevance,
                "channel": "news",
            }
        )
    return out


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _score_news_item(item: dict[str, Any]) -> float:
    relevance = item.get("relevance")
    sentiment = item.get("sentiment")
    summary_len = len(str(item.get("summary", "")))
    freshness_bonus = 0.05 if item.get("published_at") else 0.0
    score = 0.0
    if relevance is not None:
        score += float(relevance)
    if sentiment is not None:
        score += 0.15 * abs(float(sentiment))
    score += min(summary_len / 400.0, 0.10)
    score += freshness_bonus
    return score


def _score_social_item(item: dict[str, Any]) -> float:
    sentiment = item.get("sentiment")
    news_score = _score_news_item(item)
    source = str(item.get("source", "")).lower()
    socialish_bonus = 0.15 if any(k in source for k in ["social", "forum", "stocktwits", "x", "twitter", "reddit", "snowball", "eastmoney", "股吧"]) else 0.0
    sentiment_mag = abs(float(sentiment)) if sentiment is not None else 0.0
    return news_score + 0.35 * sentiment_mag + socialish_bonus


def _dedupe_articles(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (
            re.sub(r"\s+", " ", str(item.get("title", "")).strip().lower()),
            str(item.get("ticker", "") or "GLOBAL").upper(),
        )
        if not key[0] or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _detect_vendor_name(raw: Any) -> str:
    if isinstance(raw, dict) and "feed" in raw:
        return "alpha_vantage"
    if isinstance(raw, str):
        head = raw[:200].lower()
        if "source: akshare" in head:
            return "akshare"
        if "global market news" in head or "news, from" in head:
            return "yfinance_or_markdown_vendor"
    return "configured_news_vendor"


def _fetch_ticker_news(
    ts_code: str,
    start_date: str,
    end_date: str,
    alias_candidates: list[str] | None = None,
) -> tuple[list[dict[str, Any]], str, str | None]:
    vendor_input = _normalize_ts_code_for_vendor(ts_code)
    alias_candidates = [str(x).strip() for x in (alias_candidates or []) if str(x).strip()]

    try:
        raw = route_to_vendor("get_news", vendor_input, start_date, end_date)
    except Exception as primary_exc:
        primary_msg = str(primary_exc)
        fallback_items: list[dict[str, Any]] = []
        fallback_vendor = "configured_news_vendor"
        fallback_notes: list[str] = []

        if "akshare:" in primary_msg and ("empty result" in primary_msg or "No available vendor" in primary_msg):
            try:
                from dataflows.akshare_provider import get_global_news as _ak_global_news

                alias_tokens = _extract_alias_tokens_for_ts_code(ts_code)
                for alias in alias_candidates:
                    norm = _normalize_for_match(alias)
                    if norm and len(norm) >= 2 and norm not in alias_tokens:
                        alias_tokens.append(norm)
                    short = _normalize_for_match(alias[:4]) if len(alias) >= 4 else ""
                    if short and len(short) >= 2 and short not in alias_tokens and short not in BLACKLIST_GENERIC_TERMS:
                        alias_tokens.append(short)
                    short2 = _normalize_for_match(alias[:2]) if len(alias) >= 2 else ""
                    if short2 and len(short2) >= 2 and short2 not in alias_tokens and short2 not in BLACKLIST_GENERIC_TERMS:
                        alias_tokens.append(short2)

                raw_global = _ak_global_news(
                    end_date,
                    look_back_days=max(1, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days),
                    limit=50,
                )
                parsed_global = _parse_markdown_news(str(raw_global), ticker="GLOBAL")
                filtered: list[dict[str, Any]] = []
                for item in parsed_global:
                    hay = " ".join(
                        [
                            str(item.get("title", "")),
                            str(item.get("summary", "")),
                            str(item.get("source", "")),
                        ]
                    )
                    hay_norm = _normalize_for_match(hay)
                    if any(tok and tok in hay_norm for tok in alias_tokens):
                        item = dict(item)
                        item["ticker"] = ts_code
                        item.setdefault("channel", "news")
                        filtered.append(item)
                if filtered:
                    fallback_items = filtered[:10]
                    fallback_vendor = "akshare_global_keyword_fallback"
                    fallback_notes.append("ticker_news_used_global_keyword_fallback")
            except Exception as fallback_exc:
                fallback_notes.append(f"global_keyword_fallback_failed:{type(fallback_exc).__name__}:{fallback_exc}")

        if fallback_items:
            note = ";".join(fallback_notes) if fallback_notes else None
            return fallback_items, fallback_vendor, note
        raise

    vendor_name = _detect_vendor_name(raw)
    parsed = _parse_alpha_vantage_news(raw, ticker=vendor_input)
    if not parsed:
        parsed = _parse_markdown_news(str(raw), ticker=ts_code)
    for item in parsed:
        item["ticker"] = ts_code
    note = None
    if not parsed:
        note = f"no_news:{ts_code}"
    return parsed, vendor_name, note


def _fetch_global_news(trade_date: str, look_back_days: int, limit: int) -> tuple[list[dict[str, Any]], str | None]:
    raw = route_to_vendor("get_global_news", trade_date, look_back_days, limit)
    parsed = _parse_alpha_vantage_news(raw, ticker=None)
    if not parsed:
        parsed = _parse_markdown_news(str(raw), ticker="GLOBAL")
        for item in parsed:
            item["ticker"] = "GLOBAL"
    note = None if parsed else "no_global_news"
    return parsed, note


def _clean_alias_text(text: Any) -> str:
    s = str(text or "").strip()
    if not s:
        return ""
    s = re.sub(r"[（(【\[].*?[】\])）]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _normalize_for_match(text: Any) -> str:
    s = str(text or "").upper().strip()
    s = re.sub(r"[^0-9A-Z\u4e00-\u9fff]+", "", s)
    return s


def _extract_candidate_aliases(row: pd.Series) -> list[str]:
    ts_code = str(row.get("ts_code", "")).strip().upper()
    aliases: list[str] = []
    for col in ["name", "name_x", "name_y"]:
        val = _clean_alias_text(row.get(col))
        if val:
            aliases.append(val)
    aliases.extend(MANUAL_TICKER_ALIASES.get(ts_code, []))

    extra: list[str] = []
    for alias in aliases:
        a = _clean_alias_text(alias)
        if not a:
            continue
        extra.append(a)
        extra.append(a.upper())
        extra.append(a.replace("股份", ""))
        extra.append(a.replace("集团", ""))
        if len(a) >= 4:
            extra.append(a[:4])
        if len(a) >= 2 and a not in BLACKLIST_GENERIC_TERMS:
            extra.append(a[:2])
    deduped: list[str] = []
    seen: set[str] = set()
    for alias in extra:
        alias = _clean_alias_text(alias)
        norm = _normalize_for_match(alias)
        if not norm or len(norm) < 2 or norm in seen:
            continue
        seen.add(norm)
        deduped.append(alias)
    return deduped


def _extract_alias_tokens_for_ts_code(ts_code: str) -> list[str]:
    aliases = list(MANUAL_TICKER_ALIASES.get(str(ts_code).strip().upper(), []))
    tokens: list[str] = []
    seen: set[str] = set()
    for alias in aliases:
        cleaned = _clean_alias_text(alias)
        if not cleaned:
            continue
        for candidate in [cleaned, cleaned.upper(), cleaned.replace("股份", ""), cleaned.replace("集团", "")]:
            norm = _normalize_for_match(candidate)
            if norm and len(norm) >= 2 and norm not in seen:
                seen.add(norm)
                tokens.append(norm)
    return tokens


def _candidate_alias_map(candidate_pool: pd.DataFrame) -> dict[str, dict[str, Any]]:
    alias_map: dict[str, dict[str, Any]] = {}
    for _, row in candidate_pool.iterrows():
        ts_code = str(row.get("ts_code", "")).strip().upper()
        if not ts_code:
            continue
        aliases = _extract_candidate_aliases(row)
        alias_map[ts_code] = {
            "ts_code": ts_code,
            "display_name": _clean_alias_text(row.get("name") or row.get("name_x") or row.get("name_y") or ts_code),
            "aliases": aliases,
        }
    return alias_map


def _dailyhot_api_base(override: str | None = None) -> str:
    return str(override or os.getenv("DAILYHOT_API_BASE") or DEFAULT_SOCIAL_HOT_API_BASE).rstrip("/")


def _dailyhot_headers() -> dict[str, str]:
    headers = {"User-Agent": "TradingAgents2.0 social_hot_context/0.1"}
    token = os.getenv("DAILYHOT_API_TOKEN", "").strip()
    if token:
        headers["Authorization"] = token if token.lower().startswith("bearer ") else f"Bearer {token}"
    return headers


def _coerce_hot_rank(item: dict[str, Any], idx: int) -> int:
    for key in ["rank", "hot_rank", "index", "sort", "position"]:
        value = item.get(key)
        if value is None or value == "":
            continue
        try:
            return int(value)
        except Exception:
            continue
    return idx + 1


def _coerce_hot_title(item: dict[str, Any]) -> str:
    for key in ["title", "name", "desc", "text", "word", "keyword", "query"]:
        value = _clean_alias_text(item.get(key))
        if value:
            return value
    return ""


def _coerce_hot_published_at(item: dict[str, Any]) -> str:
    for key in ["published_at", "pubDate", "ctime", "mtime", "time", "timestamp", "updateTime"]:
        value = item.get(key)
        if value is None or value == "":
            continue
        return str(value)
    return ""


def _parse_tophub_rows(html: str, source: str, limit: int = 20) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    row_matches = re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.S | re.I)
    for row_html in row_matches:
        rank_match = re.search(r"<td[^>]*align=[\"']center[\"'][^>]*>\s*(\d+)\.?\s*</td>", row_html, flags=re.S | re.I)
        if not rank_match:
            continue
        rank = int(rank_match.group(1))
        anchors = re.findall(r"<a[^>]+href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", row_html, flags=re.S | re.I)
        title = ""
        url = ""
        for href, inner in anchors:
            inner_text = re.sub(r"<[^>]+>", " ", inner)
            inner_text = re.sub(r"\s+", " ", inner_text).strip()
            if inner_text and inner_text not in {"查看详细", ""} and not inner_text.startswith("http"):
                title = _clean_alias_text(inner_text)
                url = href.strip()
                break
        if not title:
            continue

        score = ""
        score_match = re.search(r'<td[^>]*class=[\"\']ws[\"\'][^>]*>(.*?)</td>', row_html, flags=re.S | re.I)
        if score_match:
            score = _clean_alias_text(re.sub(r"<[^>]+>", " ", score_match.group(1)))
        if not score:
            tds = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S | re.I)
            if len(tds) >= 3:
                score = _clean_alias_text(re.sub(r"<[^>]+>", " ", tds[-2]))

        out.append(
            {
                "source_type": source,
                "source_label": SOURCE_LABELS.get(source, source),
                "source_rank": rank,
                "title": title,
                "summary": score,
                "url": url,
                "published_at": "",
                "raw": {"html": row_html[:500]},
            }
        )
        if len(out) >= limit:
            break
    return out


def _fetch_social_hot_source(source: str, api_base: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if source not in TOPHUB_SOURCE_URLS:
        raise ValueError(f"Unsupported social hot source: {source}")
    url = TOPHUB_SOURCE_URLS[source]
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        "Referer": "https://tophub.today/",
    }
    resp = requests.get(url, headers=headers, timeout=_SOCIAL_REQUEST_TIMEOUT)
    resp.raise_for_status()
    items = _parse_tophub_rows(resp.text, source=source, limit=limit)
    if not items:
        raise ValueError(f"No hotspot rows parsed from TopHub page: {url}")
    return items


def _parse_recency_hours(value: Any, as_of: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.isdigit() and len(text) >= 10:
            ts = int(text[:10])
            dt = datetime.fromtimestamp(ts)
        else:
            cleaned = text.replace("T", " ").replace("Z", "")
            cleaned = re.sub(r"\+\d\d:?\d\d$", "", cleaned)
            dt = datetime.fromisoformat(cleaned)
        anchor = datetime.strptime(as_of, "%Y-%m-%d") + timedelta(hours=15)
        hours = (anchor - dt).total_seconds() / 3600.0
        if hours < 0:
            return 0.0
        return round(hours, 3)
    except Exception:
        return None


def _score_social_bonus(hot_mention_count: int, hot_source_count: int, hot_best_rank: int | None) -> float:
    if hot_mention_count <= 0 or hot_source_count <= 0:
        return 0.0
    bonus = 0.0
    if hot_best_rank is not None:
        if hot_best_rank <= 3:
            bonus += 0.02
        elif hot_best_rank <= 10:
            bonus += 0.01
    bonus += min(hot_source_count, 3) * 0.01
    if hot_mention_count >= 3:
        bonus += 0.01
    return round(min(bonus, 0.05), 4)


def _match_social_hot_to_candidates(
    candidate_pool: pd.DataFrame,
    hot_items: list[dict[str, Any]],
    trade_date: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    alias_map = _candidate_alias_map(candidate_pool)
    rows: list[dict[str, Any]] = []

    for ts_code, meta in alias_map.items():
        matched_items: list[dict[str, Any]] = []
        aliases = sorted(meta["aliases"], key=lambda x: len(_normalize_for_match(x)), reverse=True)
        for item in hot_items:
            text = " ".join([
                str(item.get("title", "")),
                str(item.get("summary", "")),
            ])
            norm_text = _normalize_for_match(text)
            if not norm_text:
                continue
            matched_alias = None
            for alias in aliases:
                norm_alias = _normalize_for_match(alias)
                if len(norm_alias) < 2:
                    continue
                if norm_alias in BLACKLIST_GENERIC_TERMS:
                    continue
                if norm_alias in norm_text:
                    matched_alias = alias
                    break
            if matched_alias is None:
                continue
            enriched = dict(item)
            enriched["matched_alias"] = matched_alias
            enriched["ts_code"] = ts_code
            enriched["display_name"] = meta["display_name"]
            enriched["recency_hours"] = _parse_recency_hours(item.get("published_at"), trade_date)
            matched_items.append(enriched)

        mention_count = len(matched_items)
        source_count = len({str(x.get("source_type", "")) for x in matched_items if str(x.get("source_type", ""))})
        best_rank = min([int(x.get("source_rank", 9999)) for x in matched_items], default=None)
        recencies = [x.get("recency_hours") for x in matched_items if x.get("recency_hours") is not None]
        hot_recency_hours = min(recencies) if recencies else None
        social_bonus_score = _score_social_bonus(mention_count, source_count, best_rank)
        rows.append(
            {
                "ts_code": ts_code,
                "name": meta["display_name"],
                "hot_mention_count": int(mention_count),
                "hot_source_count": int(source_count),
                "hot_best_rank": None if best_rank is None else int(best_rank),
                "hot_recency_hours": hot_recency_hours,
                "social_bonus_score": social_bonus_score,
                "matched_hot_items": matched_items[:8],
            }
        )

    rows = sorted(
        rows,
        key=lambda x: (
            -int(x.get("hot_source_count", 0)),
            -int(x.get("hot_mention_count", 0)),
            9999 if x.get("hot_best_rank") is None else int(x.get("hot_best_rank")),
            -float(x.get("social_bonus_score", 0.0)),
            str(x.get("ts_code", "")),
        ),
    )

    matched_rows = [x for x in rows if int(x.get("hot_mention_count", 0)) > 0]
    summary = {
        "enabled": True,
        "candidate_count": int(len(candidate_pool)),
        "matched_candidate_count": int(len(matched_rows)),
        "mentioned_total_count": int(sum(int(x.get("hot_mention_count", 0)) for x in matched_rows)),
        "source_coverage": dict(Counter(str(item.get("source_type", "")) for row in matched_rows for item in row.get("matched_hot_items", []))),
        "top_matches": [
            {
                "ts_code": x["ts_code"],
                "name": x["name"],
                "hot_mention_count": x["hot_mention_count"],
                "hot_source_count": x["hot_source_count"],
                "hot_best_rank": x["hot_best_rank"],
                "social_bonus_score": x["social_bonus_score"],
            }
            for x in matched_rows[:10]
        ],
    }
    return rows, summary


def _score_theme_bonus(theme_hits: list[dict[str, Any]]) -> float:
    if not theme_hits:
        return 0.0
    seen_sources = {str(x.get("source_kind", "")) for x in theme_hits if str(x.get("source_kind", ""))}
    seen_themes = {str(x.get("theme", "")) for x in theme_hits if str(x.get("theme", ""))}
    bonus = sum(THEME_SOURCE_BONUS.get(source, 0.0) for source in seen_sources)
    if len(seen_themes) >= 2:
        bonus += 0.01
    if len(theme_hits) >= 3:
        bonus += 0.01
    return round(min(bonus, 0.08), 4)


def _detect_theme_hits_from_items(items: list[dict[str, Any]], source_kind: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in items:
        text = " ".join([str(item.get("title", "")), str(item.get("summary", "")), str(item.get("source", ""))]).lower()
        if not text.strip():
            continue
        for theme, keywords in THEME_KEYWORDS.items():
            matched = [kw for kw in keywords if kw.lower() in text]
            if not matched:
                continue
            hits.append(
                {
                    "theme": theme,
                    "theme_label": THEME_LABELS.get(theme, theme),
                    "matched_keywords": matched,
                    "title": item.get("title", ""),
                    "source_kind": source_kind,
                }
            )
    return hits


def _match_theme_hot_to_candidates(
    candidate_pool: pd.DataFrame,
    news_items: list[dict[str, Any]],
    social_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    all_hits = _detect_theme_hits_from_items(news_items, "news") + _detect_theme_hits_from_items(social_items, "social")
    for _, row in candidate_pool.iterrows():
        ts_code = str(row.get("ts_code", "")).strip().upper()
        name = _clean_alias_text(row.get("name") or row.get("name_x") or row.get("name_y") or ts_code)
        industry = _clean_alias_text(row.get("industry"))
        matched_hits: list[dict[str, Any]] = []
        for hit in all_hits:
            theme = str(hit.get("theme", ""))
            if ts_code in THEME_TICKER_HINTS.get(theme, []):
                matched_hits.append(dict(hit, match_reason="ticker_hint"))
                continue
            if industry and industry in THEME_INDUSTRY_HINTS.get(theme, []):
                matched_hits.append(dict(hit, match_reason="industry_hint"))
                continue
        theme_names = sorted({str(x.get("theme_label", "")) for x in matched_hits if str(x.get("theme_label", ""))})
        source_count = len({str(x.get("source_kind", "")) for x in matched_hits if str(x.get("source_kind", ""))})
        rows.append(
            {
                "ts_code": ts_code,
                "name": name,
                "industry": industry,
                "theme_match_count": len(matched_hits),
                "theme_source_count": source_count,
                "theme_names": "|".join(theme_names),
                "theme_bonus_score": _score_theme_bonus(matched_hits),
                "matched_theme_items": matched_hits[:12],
            }
        )
    rows = sorted(rows, key=lambda x: (-int(x.get("theme_match_count", 0)), -int(x.get("theme_source_count", 0)), -float(x.get("theme_bonus_score", 0.0)), str(x.get("ts_code", ""))))
    matched_rows = [x for x in rows if int(x.get("theme_match_count", 0)) > 0]
    summary = {
        "enabled": True,
        "candidate_count": int(len(candidate_pool)),
        "matched_candidate_count": int(len(matched_rows)),
        "theme_hit_total_count": int(sum(int(x.get("theme_match_count", 0)) for x in matched_rows)),
        "top_matches": [
            {
                "ts_code": x["ts_code"],
                "name": x["name"],
                "industry": x.get("industry", ""),
                "theme_match_count": x["theme_match_count"],
                "theme_source_count": x["theme_source_count"],
                "theme_names": x["theme_names"],
                "theme_bonus_score": x["theme_bonus_score"],
            }
            for x in matched_rows[:10]
        ],
    }
    return rows, summary


def _build_proxy_ticker_news_items(
    row: pd.Series,
    hot_feature: dict[str, Any] | None,
    theme_feature: dict[str, Any] | None,
    trade_date: str,
) -> list[dict[str, Any]]:
    ts_code = str(row.get("ts_code", "")).strip().upper()
    name = _clean_alias_text(row.get("name") or row.get("name_x") or row.get("name_y") or ts_code)
    industry = _clean_alias_text(row.get("industry"))
    out: list[dict[str, Any]] = []

    if hot_feature and int(hot_feature.get("hot_mention_count", 0)) > 0:
        titles = [str(x.get("title", "")).strip() for x in hot_feature.get("matched_hot_items", []) if str(x.get("title", "")).strip()]
        out.append(
            {
                "ticker": ts_code,
                "title": f"{name} 命中社交热榜映射",
                "summary": f"{trade_date} social_hot_context 命中 {int(hot_feature.get('hot_mention_count', 0))} 次，来源 {hot_feature.get('hot_source_count', 0)} 个，最佳热榜排名 {hot_feature.get('hot_best_rank', '')}。"
                + (f" 相关热词：{' | '.join(titles[:3])}" if titles else ""),
                "source": "tophub.today",
                "link": "",
                "published_at": trade_date,
                "sentiment": None,
                "relevance": float(hot_feature.get("social_bonus_score", 0.0) or 0.0),
                "channel": "news_proxy_social_hot",
            }
        )

    if theme_feature and int(theme_feature.get("theme_match_count", 0)) > 0:
        matched_titles = [str(x.get("title", "")).strip() for x in theme_feature.get("matched_theme_items", []) if str(x.get("title", "")).strip()]
        theme_names = str(theme_feature.get("theme_names", "")).strip()
        out.append(
            {
                "ticker": ts_code,
                "title": f"{name} 命中主题新闻映射",
                "summary": f"{trade_date} theme_hot_context 命中 {int(theme_feature.get('theme_match_count', 0))} 条，行业 {industry or '未知'}，主题 {theme_names or '未标注'}。"
                + (f" 代表条目：{' | '.join(matched_titles[:3])}" if matched_titles else ""),
                "source": "global_news+theme_mapping",
                "link": "",
                "published_at": trade_date,
                "sentiment": None,
                "relevance": float(theme_feature.get("theme_bonus_score", 0.0) or 0.0),
                "channel": "news_proxy_theme_hot",
            }
        )

    return out


def build_news_social_context(
    candidate_pool: pd.DataFrame,
    trade_date: str,
    top_k_candidates: int = 15,
    news_top_n: int = 10,
    social_top_n: int = 10,
    look_back_days: int = 3,
    global_news_limit: int = 10,
    enable_social_hot_context: bool = True,
    social_hot_api_base: str | None = None,
    social_hot_sources: list[str] | None = None,
    social_hot_limit_per_source: int = 20,
    enable_xueqiu: bool = True,
    enable_twitter: bool = True,
) -> NewsContextResult:
    pool = candidate_pool.head(top_k_candidates).copy()
    start_date = (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    all_items: list[dict[str, Any]] = []
    global_items: list[dict[str, Any]] = []
    vendor = "configured_news_vendor"
    notes: list[str] = []
    social_hot_features: list[dict[str, Any]] = []
    social_hot_summary: dict[str, Any] = {"enabled": bool(enable_social_hot_context)}
    theme_hot_features: list[dict[str, Any]] = []
    theme_hot_summary: dict[str, Any] = {"enabled": True}
    ticker_fail_rows: list[tuple[pd.Series, str]] = []

    for _, row in pool.iterrows():
        ts_code = str(row.get("ts_code", "")).strip()
        if not ts_code:
            continue
        try:
            alias_candidates = [
                _clean_alias_text(row.get("name") or ""),
                _clean_alias_text(row.get("name_x") or ""),
                _clean_alias_text(row.get("name_y") or ""),
            ]
            items, vendor_name, note = _fetch_ticker_news(ts_code, start_date, trade_date, alias_candidates=alias_candidates)
        except Exception as exc:
            items, vendor_name, note = [], vendor, f"ticker_news_fetch_failed:{ts_code}:{type(exc).__name__}:{exc}"
            ticker_fail_rows.append((row.copy(), str(note)))
        if vendor == "configured_news_vendor" and vendor_name:
            vendor = vendor_name
        if note:
            notes.append(note)
        for item in items:
            item["rank_in_pool"] = int(row.get("rank_in_live_day", len(all_items) + 1)) if pd.notna(row.get("rank_in_live_day", None)) else None
            item["candidate_score"] = _to_float(row.get("overnight_live_score"))
        all_items.extend(items)

    try:
        global_items, global_note = _fetch_global_news(trade_date, look_back_days=look_back_days, limit=global_news_limit)
    except Exception as exc:
        global_items, global_note = [], f"global_news_fetch_failed:{type(exc).__name__}:{exc}"
    if global_note:
        notes.append(global_note)

    all_items = _dedupe_articles(all_items)
    global_items = _dedupe_articles(global_items)

    news_ranked = sorted(all_items, key=_score_news_item, reverse=True)[:news_top_n]
    social_ranked = sorted(all_items, key=_score_social_item, reverse=True)[:social_top_n]
    if len(news_ranked) < news_top_n and global_items:
        needed = news_top_n - len(news_ranked)
        news_ranked.extend(global_items[:needed])

    if enable_social_hot_context:
        selected_sources = social_hot_sources or DEFAULT_SOCIAL_HOT_SOURCES
        hot_items: list[dict[str, Any]] = []
        social_hot_summary = {
            "enabled": True,
            "provider": "tophub.today",
            "sources_requested": selected_sources,
            "sources_success_count": 0,
            "sources_failure_count": 0,
        }
        for source in selected_sources:
            try:
                source_items = _fetch_social_hot_source(source, api_base=social_hot_api_base, limit=social_hot_limit_per_source)
                social_hot_summary["sources_success_count"] += 1
                hot_items.extend(source_items)
            except Exception as exc:
                social_hot_summary["sources_failure_count"] += 1
                notes.append(f"social_hot_fetch_failed:{source}:{type(exc).__name__}:{exc}")
        social_hot_features, matched_summary = _match_social_hot_to_candidates(pool, hot_items, trade_date)
        social_hot_summary.update(matched_summary)
    else:
        hot_items = []
        social_hot_summary = {"enabled": False}

    theme_hot_features, theme_hot_summary = _match_theme_hot_to_candidates(pool, global_items[:global_news_limit], hot_items)

    # ── Xueqiu hot stocks & posts ──────────────────────────────────────
    xueqiu_hot_features: list[dict[str, Any]] = []
    xueqiu_hot_summary: dict[str, Any] = {"enabled": bool(enable_xueqiu)}
    if enable_xueqiu:
        xq_provider = _get_xueqiu_provider()
        if xq_provider and xq_provider is not False:
            try:
                xueqiu_result = xq_provider(pool, trade_date)
                xueqiu_hot_features = xueqiu_result.features
                xueqiu_hot_summary = xueqiu_result.summary
                if xueqiu_result.degraded:
                    notes.append(f"xueqiu:{xueqiu_result.degrade_reason}")
            except Exception as exc:
                notes.append(f"xueqiu_fetch_failed:{type(exc).__name__}:{exc}")
                xueqiu_hot_summary = {"enabled": True, "provider": "xueqiu.com", "error": str(exc)}
        else:
            xueqiu_hot_summary = {"enabled": True, "provider": "xueqiu.com", "status": "provider_not_available"}
            notes.append("xueqiu:provider_not_available")
    else:
        xueqiu_hot_summary = {"enabled": False}

    # ── Twitter sentiment ──────────────────────────────────────────────
    twitter_features: list[dict[str, Any]] = []
    twitter_summary: dict[str, Any] = {"enabled": bool(enable_twitter)}
    if enable_twitter:
        tw_provider = _get_twitter_provider()
        if tw_provider and tw_provider is not False:
            try:
                twitter_result = tw_provider(pool, trade_date)
                twitter_features = twitter_result.features
                twitter_summary = twitter_result.summary
                if twitter_result.degraded:
                    notes.append(f"twitter:{twitter_result.degrade_reason}")
            except Exception as exc:
                notes.append(f"twitter_fetch_failed:{type(exc).__name__}:{exc}")
                twitter_summary = {"enabled": True, "provider": "twitter-cli", "error": str(exc)}
        else:
            twitter_summary = {"enabled": True, "provider": "twitter-cli", "status": "provider_not_available"}
            notes.append("twitter:provider_not_available")
    else:
        twitter_summary = {"enabled": False}

    if ticker_fail_rows:
        hot_by_code = {str(x.get("ts_code", "")).strip().upper(): x for x in social_hot_features}
        theme_by_code = {str(x.get("ts_code", "")).strip().upper(): x for x in theme_hot_features}
        proxy_recovered = 0
        proxy_codes: list[str] = []
        for row, _fail_note in ticker_fail_rows:
            ts_code = str(row.get("ts_code", "")).strip().upper()
            proxy_items = _build_proxy_ticker_news_items(
                row,
                hot_by_code.get(ts_code),
                theme_by_code.get(ts_code),
                trade_date,
            )
            if not proxy_items:
                continue
            proxy_recovered += 1
            proxy_codes.append(ts_code)
            for item in proxy_items:
                item["rank_in_pool"] = int(row.get("rank_in_live_day", len(all_items) + 1)) if pd.notna(row.get("rank_in_live_day", None)) else None
                item["candidate_score"] = _to_float(row.get("overnight_live_score"))
            all_items.extend(proxy_items)
        if proxy_recovered:
            notes.append(f"ticker_news_proxy_recovered:{proxy_recovered}:{'|'.join(proxy_codes[:20])}")

    all_items = _dedupe_articles(all_items)
    global_items = _dedupe_articles(global_items)

    news_ranked = sorted(all_items, key=_score_news_item, reverse=True)[:news_top_n]
    for i, item in enumerate(social_ranked, start=1):
        item["social_rank"] = i
    for i, item in enumerate(global_items[:global_news_limit], start=1):
        item["global_rank"] = i

    return NewsContextResult(
        news_top10=news_ranked,
        social_top10=social_ranked,
        global_news_top10=global_items[:global_news_limit],
        vendor=vendor,
        notes=notes,
        social_hot_features=social_hot_features,
        social_hot_summary=social_hot_summary,
        theme_hot_features=theme_hot_features,
        theme_hot_summary=theme_hot_summary,
        xueqiu_hot_features=xueqiu_hot_features,
        xueqiu_hot_summary=xueqiu_hot_summary,
        twitter_features=twitter_features,
        twitter_summary=twitter_summary,
    )


def summarize_news_social_context(ctx: NewsContextResult) -> dict[str, Any]:
    ticker_failure_notes = [n for n in ctx.notes if str(n).startswith("ticker_news_fetch_failed:")]
    global_failure_notes = [n for n in ctx.notes if str(n).startswith("global_news_fetch_failed:")]
    no_news_notes = [n for n in ctx.notes if str(n).startswith("no_news:")]
    social_hot_failure_notes = [n for n in ctx.notes if str(n).startswith("social_hot_fetch_failed:")]
    no_global_news = any(str(n) == "no_global_news" for n in ctx.notes)

    failure_reason_counts: Counter[str] = Counter()
    for note in ticker_failure_notes + global_failure_notes + social_hot_failure_notes:
        parts = str(note).split(":", 3)
        if len(parts) >= 3:
            failure_reason_counts[parts[2]] += 1
        else:
            failure_reason_counts["unknown"] += 1

    degraded = bool(ticker_failure_notes or global_failure_notes or no_news_notes or no_global_news or social_hot_failure_notes)

    return {
        "vendor": ctx.vendor,
        "degraded": degraded,
        "degrade_reasons": sorted(set(
            (["ticker_news_fetch_failed"] if ticker_failure_notes else [])
            + (["global_news_fetch_failed"] if global_failure_notes else [])
            + (["no_ticker_news"] if no_news_notes else [])
            + (["no_global_news"] if no_global_news else [])
            + (["social_hot_fetch_failed"] if social_hot_failure_notes else [])
        )),
        "ticker_news_success_count": len(dict(Counter([str(x.get("ticker", "")) for x in ctx.news_top10 if str(x.get("ticker", "")).upper() != "GLOBAL"]))),
        "ticker_news_failure_count": len(ticker_failure_notes),
        "global_news_success_count": len(ctx.global_news_top10),
        "global_news_failure_count": len(global_failure_notes) + int(no_global_news),
        "social_hot_success_count": int(sum(1 for x in ctx.social_hot_features if int(x.get("hot_mention_count", 0)) > 0)),
        "theme_hot_success_count": int(sum(1 for x in ctx.theme_hot_features if int(x.get("theme_match_count", 0)) > 0)),
        "social_hot_failure_count": len(social_hot_failure_notes),
        "xueqiu_hot_success_count": int(sum(1 for x in ctx.xueqiu_hot_features if x.get("xq_hot_stock_rank") is not None or int(x.get("xq_hot_post_match_count", 0)) > 0)),
        "twitter_success_count": int(sum(1 for x in ctx.twitter_features if int(x.get("twitter_mention_count", 0)) > 0)),
        "failure_reason_counts": dict(failure_reason_counts),
        "news_top10_count": len(ctx.news_top10),
        "social_top10_count": len(ctx.social_top10),
        "global_news_top10_count": len(ctx.global_news_top10),
        "news_ticker_counts": dict(Counter([str(x.get("ticker", "")) for x in ctx.news_top10])),
        "social_ticker_counts": dict(Counter([str(x.get("ticker", "")) for x in ctx.social_top10])),
        "social_hot_summary": ctx.social_hot_summary,
        "theme_hot_summary": ctx.theme_hot_summary,
        "xueqiu_hot_summary": ctx.xueqiu_hot_summary,
        "twitter_summary": ctx.twitter_summary,
        "notes": ctx.notes,
    }


def build_news_social_context_block(ctx: NewsContextResult) -> str:
    payload = {
        "vendor": ctx.vendor,
        "notes": ctx.notes + [
            "social_sentiment_top10 is sentiment-style ranking derived from currently available news/aggregated sources; it is not a dedicated standalone social-media feed unless such a provider is later added.",
            "social_hot_context is a lightweight hotlist-to-ticker mapping layer over recall TopK candidates. Use it as a soft signal only, not as a hard screening rule.",
            "xueqiu_hot_context comes from xueqiu.com (雪球) hot stocks ranking and hot posts matching against candidates. Treat as retail sentiment signal.",
            "twitter_context comes from Twitter/X A-stock market keyword searches matched against candidates. Treat as global sentiment signal; note that twitter-cli may not always be available.",
        ],
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
        "xueqiu_hot_context": {
            "summary": ctx.xueqiu_hot_summary,
            "features": ctx.xueqiu_hot_features,
        },
        "twitter_context": {
            "summary": ctx.twitter_summary,
            "features": ctx.twitter_features,
        },
    }
    return (
        "\n\n附加上下文：以下是程序预抓取并排序后的 News Top10 / Social-sentiment Top10 / Social Hot Context。\n"
        "使用规则：\n"
        "- 可以引用这些条目作为风险/催化/情绪依据。\n"
        "- 不要声称 social_sentiment_top10 来自独立 Twitter/Reddit/雪球 专用接口，除非条目 source 明确显示。\n"
        "- social_hot_context 只表示热点榜单与 recall 候选的关键词映射结果，适合作为 soft signal。\n"
        "- theme_hot_context 表示主题→行业→个股 的弱监督映射结果，适合作为板块/语义 soft signal。\n"
        "- 如条目不足、为空或 source 含糊，宁可保守表述。\n"
        "```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
