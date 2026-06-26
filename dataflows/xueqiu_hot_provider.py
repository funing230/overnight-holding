# -*- coding: utf-8 -*-
"""Xueqiu (雪球) hot stocks & hot posts provider for overnight live context.

This module wraps the Xueqiu HTTP API to fetch:
- Hot stocks ranking (人气榜 / 关注榜)
- Hot posts from the public timeline

Authentication is optional: public endpoints may work with a homepage-visited
session cookie (acw_tc anti-DDoS).  If a stronger cookie (xq_a_token) is
available, authenticated endpoints return richer data.

All fetchers are defensive — failures return empty structures instead of raising.
"""

from __future__ import annotations

import http.cookiejar
import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_REFERER = "https://api.xueqiu.com/"
_XUEQIU_HOME = "https://api.xueqiu.com"
_TIMEOUT = 12

_XUEQIU_HOT_STOCKS_URL = "https://stock.xueqiu.com/v5/stock/hot_stock/list.json"
_XUEQIU_HOT_POSTS_URL = (
    "https://api.xueqiu.com/v4/statuses/public_timeline_by_category.json"
    "?since_id=-1&max_id=-1&count=20&category=-1"
)
# Alternative: hot-list endpoint (richer engagement data)
_XUEQIU_HOT_LIST_URL = (
    "https://api.xueqiu.com/statuses/hot/listV2.json"
    "?type=10&count=20"
)
_XUEQIU_STOCK_QUOTE_URL = "https://stock.xueqiu.com/v5/stock/batch/quote.json"


@dataclass
class XueqiuHotResult:
    hot_stocks: list[dict[str, Any]] = field(default_factory=list)
    hot_posts: list[dict[str, Any]] = field(default_factory=list)
    features: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    degraded: bool = True
    degrade_reason: str = ""


# ── Cookie helpers ────────────────────────────────────────────────────────

_cookie_jar = http.cookiejar.CookieJar()
_opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(_cookie_jar),
)
_cookies_seeded = False


def _inject_cookie_string(cookie_str: str) -> None:
    for pair in cookie_str.split(";"):
        pair = pair.strip()
        if "=" not in pair:
            continue
        name, _, value = pair.partition("=")
        cookie = http.cookiejar.Cookie(
            version=0,
            name=name.strip(),
            value=value.strip(),
            port=None,
            port_specified=False,
            domain=".xueqiu.com",
            domain_specified=True,
            domain_initial_dot=True,
            path="/",
            path_specified=True,
            secure=True,
            expires=None,
            discard=True,
            comment=None,
            comment_url=None,
            rest={},
        )
        _cookie_jar.set_cookie(cookie)


def _seed_cookies() -> None:
    """Best-effort cookie seeding.  Never raises."""
    global _cookies_seeded
    if _cookies_seeded:
        return
    _cookies_seeded = True

    # 1) Explicit env var
    env_cookie = os.getenv("XUEQIU_COOKIE", "")
    if env_cookie:
        _inject_cookie_string(env_cookie)
        return

    # 2) Homepage visit for acw_tc anti-DDoS cookie
    try:
        req = urllib.request.Request(
            _XUEQIU_HOME, headers={"User-Agent": _UA}
        )
        _opener.open(req, timeout=_TIMEOUT)
    except Exception:
        pass


def _get_json(url: str) -> Any:
    _seed_cookies()
    req = urllib.request.Request(
        url, headers={"User-Agent": _UA, "Referer": _REFERER}
    )
    with _opener.open(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">")):
        text = text.replace(entity, char)
    return text.strip()


# ── Fetchers ──────────────────────────────────────────────────────────────


def fetch_xueqiu_hot_stocks(limit: int = 15, stock_type: int = 10) -> list[dict[str, Any]]:
    """Fetch Xueqiu hot stock ranking.

    Args:
        limit: Max results.
        stock_type: 10=人气榜 (default), 12=关注榜.

    Returns list of dicts with keys: symbol, name, current, percent, rank.
    """
    try:
        data = _get_json(
            f"{_XUEQIU_HOT_STOCKS_URL}?size={limit}&type={stock_type}"
        )
        items = (data.get("data") or {}).get("items") or []
        results = []
        for idx, item in enumerate(items[:limit], 1):
            results.append({
                "symbol": item.get("code") or item.get("symbol", ""),
                "name": item.get("name", ""),
                "current": item.get("current"),
                "percent": item.get("percent"),
                "rank": idx,
            })
        return results
    except Exception:
        return []


def fetch_xueqiu_hot_posts(limit: int = 20) -> list[dict[str, Any]]:
    """Fetch Xueqiu public timeline hot posts.

    Returns list of dicts with keys: id, title, text, author, likes, url.
    """
    try:
        data = _get_json(_XUEQIU_HOT_POSTS_URL)
        items = data.get("list") or []
        results = []
        for item in items[:limit]:
            try:
                post = (
                    json.loads(item["data"])
                    if isinstance(item.get("data"), str)
                    else {}
                )
            except (json.JSONDecodeError, KeyError):
                post = {}
            user = post.get("user") or {}
            text = _strip_html(
                post.get("text") or post.get("description") or ""
            )
            target = post.get("target", "")
            results.append({
                "id": post.get("id", 0),
                "title": post.get("title") or "",
                "text": text[:300],
                "author": user.get("screen_name", ""),
                "likes": post.get("like_count", 0),
                "url": f"https://xueqiu.com{target}" if target else "",
            })
        return results
    except Exception:
        return []


# ── Feature mapping ───────────────────────────────────────────────────────


# Known Xueqiu symbol → ts_code mapping patterns.
# Xueqiu uses SH600519 / SZ000858 / BJ... format.
def _xq_symbol_to_ts_code(symbol: str) -> str | None:
    """Convert Xueqiu symbol (SH600519) to ts_code (600519.SH)."""
    sym = str(symbol).strip().upper()
    m = re.match(r"^(SH|SZ|BJ)(\d{6})$", sym)
    if not m:
        return None
    return f"{m.group(2)}.{m.group(1)}"


def _normalize_for_match(text: str) -> str:
    s = str(text).upper().strip()
    s = re.sub(r"[^0-9A-Z\u4e00-\u9fff]+", "", s)
    return s


def _extract_name_tokens(name: str) -> list[str]:
    """Extract keyword tokens from a stock name for fuzzy matching."""
    n = str(name).strip()
    tokens = [n]
    # Remove suffix words
    for suffix in ["股份", "集团", "有限", "公司", "控股", "科技"]:
        shortened = n.replace(suffix, "")
        if len(shortened) >= 2 and shortened not in tokens:
            tokens.append(shortened)
    # Take first 2-3 chars as abbreviation
    if len(n) >= 4:
        tokens.append(n[:3])
    if len(n) >= 3:
        tokens.append(n[:2])
    return list(set(t for t in tokens if len(t) >= 2))


def match_xueqiu_hot_to_candidates(
    candidate_pool: pd.DataFrame,
    hot_stocks: list[dict[str, Any]],
    hot_posts: list[dict[str, Any]],
    trade_date: str,
) -> XueqiuHotResult:
    """Match Xueqiu hot stocks and posts against the candidate pool.

    Returns a XueqiuHotResult with per-candidate features and summary.
    """
    result = XueqiuHotResult(
        hot_stocks=hot_stocks,
        hot_posts=hot_posts,
        degraded=False,
    )

    if not hot_stocks and not hot_posts:
        result.degraded = True
        result.degrade_reason = "xueqiu_no_data"
        result.summary = {
            "enabled": True,
            "provider": "xueqiu.com",
            "hot_stocks_count": 0,
            "hot_posts_count": 0,
            "matched_candidates": 0,
        }
        return result

    pool = candidate_pool.head(30).copy()
    if "ts_code" not in pool.columns:
        result.degraded = True
        result.degrade_reason = "candidate_pool_missing_ts_code"
        result.summary = {"enabled": True, "provider": "xueqiu.com"}
        return result

    # Build lookup: ts_code → {name tokens, industry, etc.}
    candidates: dict[str, dict[str, Any]] = {}
    for _, row in pool.iterrows():
        ts_code = str(row.get("ts_code", "")).strip().upper()
        if not ts_code:
            continue
        name = str(row.get("name") or row.get("name_x") or row.get("name_y") or "")
        clean_name = name.replace(" ", "")
        candidates[ts_code] = {
            "ts_code": ts_code,
            "name": clean_name,
            "name_tokens": _extract_name_tokens(clean_name),
            "industry": str(row.get("industry", "")).strip(),
            "xq_hot_stock_rank": None,
            "xq_hot_stock_percent": None,
            "xq_hot_post_match_count": 0,
            "xq_hot_post_titles": [],
        }

    # Match hot stocks by symbol
    matched_stock_count = 0
    for stock in hot_stocks:
        ts_code = _xq_symbol_to_ts_code(stock.get("symbol", ""))
        if ts_code and ts_code in candidates:
            candidates[ts_code]["xq_hot_stock_rank"] = stock.get("rank")
            candidates[ts_code]["xq_hot_stock_percent"] = stock.get("percent")
            matched_stock_count += 1

    # Match hot posts by keyword against stock names
    matched_post_count = 0
    for post in hot_posts:
        post_text = _normalize_for_match(
            f"{post.get('title', '')} {post.get('text', '')}"
        )
        for ts_code, info in candidates.items():
            for token in info["name_tokens"]:
                token_norm = _normalize_for_match(token)
                if token_norm and len(token_norm) >= 2 and token_norm in post_text:
                    info["xq_hot_post_match_count"] += 1
                    title = post.get("title", "") or post.get("text", "")[:60]
                    if title and title not in info["xq_hot_post_titles"]:
                        info["xq_hot_post_titles"].append(title)
                    matched_post_count += 1
                    break

    # Build feature rows
    features = []
    for ts_code, info in candidates.items():
        post_count = info["xq_hot_post_match_count"]
        hot_rank = info["xq_hot_stock_rank"]
        hot_percent = info.get("xq_hot_stock_percent")

        # Bonus: stock is on Xueqiu hot list or mentioned in hot posts
        xq_bonus = 0.0
        if hot_rank is not None and isinstance(hot_rank, (int, float)):
            # Higher rank (closer to #1) = more bonus
            xq_bonus += max(0, 0.04 * (1.0 - float(hot_rank) / 20.0))
        if post_count > 0:
            xq_bonus += min(0.04, 0.01 * post_count)
        xq_bonus = round(min(xq_bonus, 0.06), 4)

        features.append({
            "ts_code": ts_code,
            "xq_hot_stock_rank": hot_rank,
            "xq_hot_stock_percent": _safe_float(hot_percent),
            "xq_hot_post_match_count": post_count,
            "xq_hot_post_titles": ";".join(info["xq_hot_post_titles"][:3]),
            "xueqiu_bonus_score": xq_bonus,
        })

    result.features = features

    # Summary
    result.summary = {
        "enabled": True,
        "provider": "xueqiu.com",
        "hot_stocks_count": len(hot_stocks),
        "hot_posts_count": len(hot_posts),
        "matched_hot_stock_candidates": matched_stock_count,
        "matched_hot_post_candidates": matched_post_count,
        "unique_candidates_with_any_match": int(
            sum(1 for f in features if f["xq_hot_stock_rank"] is not None or f["xq_hot_post_match_count"] > 0)
        ),
    }

    return result


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (ValueError, TypeError):
        return None


def _summarize_xueqiu_result(texts: list[str], limit: int = 5) -> str:
    """Create a compact text summary of hot posts mentioning matched stocks."""
    if not texts:
        return ""
    return "; ".join(texts[:limit])[:500]


def build_xueqiu_context(
    candidate_pool: pd.DataFrame,
    trade_date: str,
    hot_stock_limit: int = 15,
    hot_post_limit: int = 20,
) -> XueqiuHotResult:
    """Main entry: fetch Xueqiu hot content and match against candidates."""
    started = time.time()
    hot_stocks = fetch_xueqiu_hot_stocks(limit=hot_stock_limit)
    hot_posts = fetch_xueqiu_hot_posts(limit=hot_post_limit)
    result = match_xueqiu_hot_to_candidates(candidate_pool, hot_stocks, hot_posts, trade_date)
    elapsed = round(time.time() - started, 3)
    result.summary["elapsed_seconds"] = elapsed
    result.summary["trade_date"] = trade_date
    return result
