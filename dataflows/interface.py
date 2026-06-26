from typing import Annotated
import re
import logging
import pandas as pd
from yfinance.exceptions import YFRateLimitError

logger = logging.getLogger(__name__)

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage_common import AlphaVantageRateLimitError
from .tushare_provider import (
    get_stock_data as get_tushare_stock_data,
    get_indicators as get_tushare_indicators,
    get_fundamentals as get_tushare_fundamentals,
    get_balance_sheet as get_tushare_balance_sheet,
    get_cashflow as get_tushare_cashflow,
    get_income_statement as get_tushare_income_statement,
    get_news as get_tushare_news,
    get_global_news as get_tushare_global_news,
    get_insider_transactions as get_tushare_insider_transactions,
    TushareRateLimitError,
)
from .akshare_provider import (
    get_stock_data as get_akshare_stock_data,
    get_indicators as get_akshare_indicators,
    get_fundamentals as get_akshare_fundamentals,
    get_balance_sheet as get_akshare_balance_sheet,
    get_cashflow as get_akshare_cashflow,
    get_income_statement as get_akshare_income_statement,
    get_news as get_akshare_news,
    get_global_news as get_akshare_global_news,
    get_insider_transactions as get_akshare_insider_transactions,
    AkShareRateLimitError,
)
from .overnight_pipeline_provider import (
    get_trade_date_candidates as get_overnight_candidates_local,
    summarize_trade_date_candidates as get_overnight_candidate_summary_local,
    build_candidate_prompt_payload as get_overnight_candidate_payload_local,
)

from .resilience import (
    RateLimitError,
    DataSourceError,
    yfinance_limiter,
    alpha_vantage_limiter,
    tushare_limiter,
    akshare_limiter,
    load_file_cache,
    save_file_cache,
)

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    },
    "overnight_data": {
        "description": "Overnight strategy candidates and summaries",
        "tools": [
            "get_overnight_candidates",
            "get_overnight_candidate_summary",
            "get_overnight_candidate_payload",
        ]
    }
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "tushare",
    "akshare",
    "local",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
        "tushare": get_tushare_stock_data,
        "akshare": get_akshare_stock_data,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
        "tushare": get_tushare_indicators,
        "akshare": get_akshare_indicators,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
        "tushare": get_tushare_fundamentals,
        "akshare": get_akshare_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
        "tushare": get_tushare_balance_sheet,
        "akshare": get_akshare_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
        "tushare": get_tushare_cashflow,
        "akshare": get_akshare_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
        "tushare": get_tushare_income_statement,
        "akshare": get_akshare_income_statement,
    },
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
        "tushare": get_tushare_news,
        "akshare": get_akshare_news,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
        "tushare": get_tushare_global_news,
        "akshare": get_akshare_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "tushare": get_tushare_insider_transactions,
        "akshare": get_akshare_insider_transactions,
    },
    # overnight_data
    "get_overnight_candidates": {
        "local": get_overnight_candidates_local,
    },
    "get_overnight_candidate_summary": {
        "local": get_overnight_candidate_summary_local,
    },
    "get_overnight_candidate_payload": {
        "local": get_overnight_candidate_payload_local,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

# Per-vendor rate limiters
VENDOR_LIMITERS = {
    "yfinance": yfinance_limiter,
    "alpha_vantage": alpha_vantage_limiter,
    "tushare": tushare_limiter,
    "akshare": akshare_limiter,
}

LOCAL_ONLY_METHODS = {
    "get_overnight_candidates",
    "get_overnight_candidate_summary",
    "get_overnight_candidate_payload",
}

# ---------------------------------------------------------------------------
# Market detection and vendor compatibility
# ---------------------------------------------------------------------------

# Which markets each vendor supports
VENDOR_MARKETS = {
    "tushare":       {"cn"},
    "akshare":       {"cn"},
    "yfinance":      {"us", "global"},
    "alpha_vantage": {"us", "global"},
}

# Methods that are market-agnostic (no ticker argument, or global scope)
_MARKET_AGNOSTIC_METHODS = {"get_global_news"}


def detect_market(symbol: str) -> str:
    """Detect which market a ticker symbol belongs to.

    Returns:
        "cn"      — Chinese A-share (000001.SZ, 600000.SH, 830001.BJ, pure 6-digit)
        "us"      — US equity (AAPL, MSFT, BRK.B, ^GSPC)
        "global"  — Index or unknown (compatible with all vendors)
    """
    if not symbol:
        return "global"

    s = symbol.strip().upper()

    # Chinese A-share: 6 digits with exchange suffix
    if re.match(r"^\d{6}\.(SZ|SH|BJ)$", s):
        return "cn"

    # Pure 6-digit code → A-share
    if re.match(r"^\d{6}$", s):
        return "cn"

    # US index symbols: ^GSPC, ^IXIC, ^DJI
    if s.startswith("^"):
        return "us"

    # Pure letters (with optional dot for BRK.B style) → US
    if re.match(r"^[A-Z]{1,5}(\.[A-Z])?$", s):
        return "us"

    # ETF-style: SPY, QQQ, etc.
    if re.match(r"^[A-Z]{2,5}$", s):
        return "us"

    return "global"


def _is_empty_result(result) -> bool:
    """Check if a vendor returned a 'fake success' (data is effectively empty).

    Catches cases like yfinance returning 'No data found for symbol...'
    when given an A-share ticker it doesn't recognize.
    """
    if result is None:
        return True

    if isinstance(result, str):
        lower = result.lower()
        # Common empty-data messages
        if any(phrase in lower for phrase in (
            "no data found",
            "no records",
            "no news found",
            "no insider",
            "no fundamentals",
            "no balance sheet",
            "no cashflow",
            "no income statement",
            "error retrieving",
            "error getting",
        )):
            return True

        # CSV with only header, no data rows
        lines = [l for l in result.strip().split('\n') if l and not l.startswith('#')]
        if len(lines) <= 1:
            return True

    return False

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support.

    Features:
      - Market-aware vendor filtering (A-share tickers → CN vendors only, etc.)
      - Per-vendor rate limiting (throttle before each call)
      - Empty result detection (prevents 'fake success' from wrong vendor)
      - Automatic fallback on rate-limit, transient errors, or empty results
      - File-based parquet cache for OHLCV data (get_stock_data)
      - Local overnight-data routing for graph-ready candidate access
    """
    if method in LOCAL_ONLY_METHODS:
        if method not in VENDOR_METHODS:
            raise ValueError(f"Method '{method}' not supported")
        impl = VENDOR_METHODS[method].get("local")
        if impl is None:
            raise RuntimeError(f"Local implementation missing for '{method}'")
        return impl(*args, **kwargs)

    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    # Market-aware filtering: exclude vendors that can't handle this ticker
    if method not in _MARKET_AGNOSTIC_METHODS and args:
        market = detect_market(str(args[0]))
        if market != "global":
            compatible = [
                v for v in fallback_vendors
                if market in VENDOR_MARKETS.get(v, {"us", "cn", "global"})
            ]
            if compatible:
                if set(compatible) != set(fallback_vendors):
                    logger.debug(
                        "Market filter: %s → %s, vendors %s → %s",
                        args[0], market, fallback_vendors, compatible,
                    )
                fallback_vendors = compatible
            # If no compatible vendor, keep all (better to try than to fail)

    errors = []
    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        # Check file cache for stock data
        if method == "get_stock_data" and len(args) >= 3:
            cached = load_file_cache(vendor, args[0], args[1], args[2])
            if cached is not None and not cached.empty:
                return cached.to_csv()

        # Apply rate limiting
        limiter = VENDOR_LIMITERS.get(vendor)
        if limiter:
            limiter.wait()

        try:
            result = impl_func(*args, **kwargs)

            # Check for empty/fake-success results
            if _is_empty_result(result):
                errors.append(f"{vendor}: empty result")
                logger.debug("Empty result from %s for %s, trying next vendor", vendor, method)
                continue

            # Cache stock data results
            if method == "get_stock_data" and len(args) >= 3 and isinstance(result, str):
                try:
                    import io
                    # Skip header lines starting with #
                    lines = result.split('\n')
                    csv_start = 0
                    for i, line in enumerate(lines):
                        if line and not line.startswith('#'):
                            csv_start = i
                            break
                    csv_data = '\n'.join(lines[csv_start:])
                    df = pd.read_csv(io.StringIO(csv_data))
                    if not df.empty:
                        save_file_cache(df, vendor, args[0], args[1], args[2])
                except Exception:
                    pass  # Cache save failure is non-fatal

            return result
        except (AlphaVantageRateLimitError, TushareRateLimitError,
                AkShareRateLimitError, RateLimitError, YFRateLimitError):
            errors.append(f"{vendor}: rate limited")
            continue  # Rate limits trigger fallback
        except Exception as e:
            # For transient connection errors, also try next vendor
            err_msg = str(e).lower()
            if any(kw in err_msg for kw in (
                "connection", "timeout", "remote", "reset", "refused",
                "too many requests", "rate limit",
            )):
                errors.append(f"{vendor}: {e}")
                continue
            raise  # Non-transient errors propagate immediately

    raise RuntimeError(
        f"No available vendor for '{method}'. Errors: {' | '.join(errors)}"
    )
