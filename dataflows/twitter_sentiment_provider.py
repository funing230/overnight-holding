# -*- coding: utf-8 -*-
"""Twitter/X sentiment provider for overnight live context.

Wraps the ``twitter-cli`` CLI tool to search for A-stock-related keywords
and match results against the overnight candidate pool.  Falls back
gracefully to empty results if the CLI is not installed or authenticated.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


_TIMEOUT_S = 20

# A-stock market keywords for Twitter search — Chinese + pinyin
_MARKET_KEYWORDS_CHINESE = [
    "A股",
    "沪深300",
    "上证指数",
    "中国股市",
    "创业板",
    "科创板",
]

# CSI300 heavyweight ticker-related keywords (pinyin / English brand names)
_MARKET_KEYWORDS_GLOBAL = [
    "Moutai",
    "Kweichow",
    "CATL",
    "BYD",
    "China stock",
    "SSE Composite",
    "CSI300",
    "Shanghai Composite",
]


@dataclass
class TwitterSentimentResult:
    tweets: list[dict[str, Any]] = field(default_factory=list)
    features: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    degraded: bool = True
    degrade_reason: str = ""


def _twitter_cli_available() -> bool:
    """Check if twitter-cli is installed and authenticated."""
    try:
        proc = subprocess.run(
            ["twitter", "status"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        return proc.returncode == 0 and "ok: true" in proc.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _twitter_search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Execute a twitter-cli search and parse results.

    twitter-cli 0.8.x outputs a single JSON object with a ``data`` array.
    Older versions output newline-delimited JSON lines.  This function
    handles both formats.
    """
    cmd = ["twitter", "search", query, "--json"]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    if proc.returncode != 0:
        return []

    raw = proc.stdout.strip()
    if not raw:
        return []

    # Try single JSON object format (twitter-cli 0.8.x)
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and data.get("ok"):
            items = data.get("data", [])
            tweets = []
            for item in items[:limit]:
                author = item.get("author") or {}
                metrics = item.get("metrics") or {}
                tweets.append({
                    "id": item.get("id", ""),
                    "text": item.get("text", ""),
                    "author": author.get("screenName") or author.get("screen_name", ""),
                    "likes": metrics.get("likes", 0),
                    "retweets": metrics.get("retweets", 0),
                    "created_at": item.get("createdAt") or item.get("created_at", ""),
                    "url": f"https://x.com/{author.get('screenName', '')}/status/{item.get('id', '')}" if item.get("id") else "",
                    "query": query,
                })
            return tweets
    except (json.JSONDecodeError, KeyError, TypeError):
        pass

    # Try newline-delimited JSON format (older versions)
    tweets = []
    for line in raw.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        tweets.append({
            "id": data.get("id") or data.get("tweet_id", ""),
            "text": data.get("text") or data.get("full_text", ""),
            "author": data.get("author") or data.get("screen_name", ""),
            "likes": data.get("likes") or data.get("favorite_count", 0),
            "retweets": data.get("retweets") or data.get("retweet_count", 0),
            "created_at": data.get("created_at", ""),
            "url": data.get("url", ""),
            "query": query,
        })
    return tweets


def fetch_twitter_market_sentiment(
    limit_per_query: int = 10,
) -> list[dict[str, Any]]:
    """Search Twitter for A-stock market keywords.

    Uses Chinese keywords first, then falls back to global English keywords
    if the Chinese results are sparse.

    Returns deduplicated list of tweet dicts.
    """
    if not _twitter_cli_available():
        return []

    all_tweets: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    # Chinese keywords first
    for kw in _MARKET_KEYWORDS_CHINESE:
        try:
            results = _twitter_search(kw, limit=limit_per_query)
            for t in results:
                tid = str(t.get("id", ""))
                if tid and tid not in seen_ids:
                    seen_ids.add(tid)
                    all_tweets.append(t)
        except Exception:
            continue
        time.sleep(0.3)

    # If we got fewer than 5 results, try global keywords too
    if len(all_tweets) < 5:
        for kw in _MARKET_KEYWORDS_GLOBAL:
            try:
                results = _twitter_search(kw, limit=max(3, limit_per_query // 2))
                for t in results:
                    tid = str(t.get("id", ""))
                    if tid and tid not in seen_ids:
                        seen_ids.add(tid)
                        all_tweets.append(t)
            except Exception:
                continue
            time.sleep(0.3)

    return all_tweets


def _normalize_for_match(text: str) -> str:
    s = str(text).upper().strip()
    s = re.sub(r"[^0-9A-Z\u4e00-\u9fff]+", "", s)
    return s


def _extract_name_tokens(name: str) -> list[str]:
    """Extract keyword tokens from a stock name for fuzzy matching."""
    n = str(name).strip()
    tokens = [n]
    for suffix in ["股份", "集团", "有限", "公司", "控股", "科技", "银行", "证券", "保险"]:
        shortened = n.replace(suffix, "")
        if len(shortened) >= 2 and shortened not in tokens:
            tokens.append(shortened)
    if len(n) >= 4:
        tokens.append(n[:3])
    if len(n) >= 3:
        tokens.append(n[:2])
    return list(set(t for t in tokens if len(t) >= 2))


def match_twitter_to_candidates(
    candidate_pool: pd.DataFrame,
    tweets: list[dict[str, Any]],
    trade_date: str,
) -> TwitterSentimentResult:
    """Match Twitter A-stock market tweets against the candidate pool."""
    result = TwitterSentimentResult(tweets=tweets, degraded=False)

    if not tweets:
        result.degraded = True
        result.degrade_reason = "twitter_no_cli_or_no_results"
        result.summary = {
            "enabled": True,
            "provider": "twitter-cli",
            "tweets_fetched": 0,
            "matched_candidates": 0,
        }
        return result

    pool = candidate_pool.head(30).copy()
    if "ts_code" not in pool.columns:
        result.degraded = True
        result.degrade_reason = "candidate_pool_missing_ts_code"
        result.summary = {"enabled": True, "provider": "twitter-cli"}
        return result

    # Build candidate lookup
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
            "twitter_mention_count": 0,
            "twitter_total_likes": 0,
            "twitter_total_retweets": 0,
            "twitter_sample_texts": [],
        }

    # Match tweets against stock names
    matched_count = 0
    for tweet in tweets:
        tweet_text = _normalize_for_match(
            f"{tweet.get('text', '')} {tweet.get('query', '')}"
        )
        for ts_code, info in candidates.items():
            for token in info["name_tokens"]:
                token_norm = _normalize_for_match(token)
                if token_norm and len(token_norm) >= 2 and token_norm in tweet_text:
                    info["twitter_mention_count"] += 1
                    info["twitter_total_likes"] += int(tweet.get("likes", 0) or 0)
                    info["twitter_total_retweets"] += int(tweet.get("retweets", 0) or 0)
                    snippet = tweet.get("text", "")[:80]
                    if snippet and snippet not in info["twitter_sample_texts"]:
                        info["twitter_sample_texts"].append(snippet)
                    matched_count += 1
                    break

    # Build feature rows
    features = []
    for ts_code, info in candidates.items():
        mention_count = info["twitter_mention_count"]
        tw_bonus = 0.0
        if mention_count > 0:
            tw_bonus = min(0.04, 0.008 * mention_count + 0.002 * info["twitter_total_likes"] / max(1, mention_count))
        tw_bonus = round(tw_bonus, 4)

        features.append({
            "ts_code": ts_code,
            "twitter_mention_count": mention_count,
            "twitter_total_likes": info["twitter_total_likes"],
            "twitter_total_retweets": info["twitter_total_retweets"],
            "twitter_sample_texts": ";".join(info["twitter_sample_texts"][:2]),
            "twitter_bonus_score": tw_bonus,
        })

    result.features = features
    result.summary = {
        "enabled": True,
        "provider": "twitter-cli",
        "tweets_fetched": len(tweets),
        "matched_candidates": matched_count,
    }
    return result


def build_twitter_context(
    candidate_pool: pd.DataFrame,
    trade_date: str,
) -> TwitterSentimentResult:
    """Main entry: fetch Twitter A-stock sentiment and match against candidates."""
    started = time.time()
    tweets = fetch_twitter_market_sentiment()
    result = match_twitter_to_candidates(candidate_pool, tweets, trade_date)
    elapsed = round(time.time() - started, 3)
    result.summary["elapsed_seconds"] = elapsed
    result.summary["trade_date"] = trade_date
    return result
