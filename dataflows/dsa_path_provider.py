"""
DSA Path B bridge — daily_stock_analysis 5-Agent integration.

Calls daily_stock_analysis's multi-agent system (intel/technical/risk/
decision/portfolio) on the Selector Top15 to produce an independent
scoring dimension.

If DSA dependencies are unavailable, returns neutral scores gracefully.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DSA_ROOT = "/home/sun/.openclaw/workspace-research-main/daily_stock_analysis"

# Sentinel to avoid repeated import failures
_DSA_AVAILABLE: bool | None = None  # None = not checked yet


def _check_dsa() -> bool:
    """Check if daily_stock_analysis can be imported."""
    global _DSA_AVAILABLE
    if _DSA_AVAILABLE is not None:
        return _DSA_AVAILABLE
    try:
        import sys
        if DSA_ROOT not in sys.path:
            sys.path.insert(0, DSA_ROOT)
        from analyzer_service import analyze_stock  # noqa: F401
        _DSA_AVAILABLE = True
    except Exception as e:
        logger.warning("DSA not available: %s", e)
        _DSA_AVAILABLE = False
    return _DSA_AVAILABLE


def _normalize_code(ts_code: str) -> str:
    """Convert '600519.SH' → '600519'."""
    return ts_code.split(".", 1)[0]


def _normalize_score(sentiment_score: int) -> float:
    """Convert DSA sentiment (0-100) to 0-1 scale."""
    return max(0.0, min(1.0, sentiment_score / 100.0))


def run_dsa_analysis(
    ts_codes: list[str],
    trade_date: str,
) -> pd.DataFrame:
    """Run daily_stock_analysis multi-agent pipeline on a list of stocks.

    Calls analyze_stocks() for each code, extracts sentiment_score,
    operation_advice, confidence_level, and risk_warnings.

    Returns DataFrame with ts_code, dsa_score, dsa_operation, dsa_confidence,
    dsa_risk, dsa_available (bool). If DSA is unavailable, returns neutral scores.
    """
    if not _check_dsa():
        return _neutral_result(ts_codes, available=False)

    import sys
    if DSA_ROOT not in sys.path:
        sys.path.insert(0, DSA_ROOT)
    from analyzer_service import analyze_stock

    rows = []
    for ts_code in ts_codes:
        bare = _normalize_code(ts_code)
        try:
            result = analyze_stock(bare, full_report=False)
            if result and result.success:
                rows.append({
                    "ts_code": ts_code,
                    "dsa_score": _normalize_score(result.sentiment_score),
                    "dsa_operation": result.operation_advice or "hold",
                    "dsa_confidence": result.confidence_level or "中",
                    "dsa_risk": result.risk_warning or "",
                    "dsa_trend": result.trend_prediction or "",
                    "dsa_available": True,
                })
            else:
                rows.append(_neutral_row(ts_code))
        except Exception as e:
            logger.warning("DSA analysis failed for %s: %s", ts_code, e)
            rows.append(_neutral_row(ts_code))

    if not rows:
        return _neutral_result(ts_codes, available=False)
    return pd.DataFrame(rows)


def _neutral_row(ts_code: str) -> dict[str, Any]:
    return {
        "ts_code": ts_code,
        "dsa_score": 0.5,
        "dsa_operation": "hold",
        "dsa_confidence": "低",
        "dsa_risk": "",
        "dsa_trend": "震荡",
        "dsa_available": False,
    }


def _neutral_result(ts_codes: list[str], available: bool = False) -> pd.DataFrame:
    return pd.DataFrame([_neutral_row(c) for c in ts_codes]).assign(
        dsa_available=available
    )
