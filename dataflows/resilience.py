"""
Shared resilience utilities for all data vendors.

Provides:
  - SimpleRateLimiter: per-vendor request throttling
  - File-based parquet cache for OHLCV data
  - HTTP session with SQLite-backed persistent cache
  - Retry decorator with exponential backoff
  - Unified exception hierarchy
"""

from __future__ import annotations

import hashlib
import time
import threading
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Cache directories
# ---------------------------------------------------------------------------

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FILE_CACHE_DIR = CACHE_DIR / "bars"
FILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Trading hours detection
# ---------------------------------------------------------------------------


def is_trading_hours() -> bool:
    """Check if current time is within trading hours (CN or US).

    CN: 09:15 - 15:30 CST (UTC+8), Mon-Fri
    US: 09:00 - 16:30 EST (UTC-5), Mon-Fri
    Conservative: if either market could be open, return True.
    """
    from datetime import timezone
    now_utc = datetime.now(timezone.utc)
    weekday = now_utc.weekday()
    if weekday >= 5:  # Weekend
        return False
    hour_utc = now_utc.hour
    # CN market: ~01:15 - 07:30 UTC
    # US market: ~14:00 - 21:30 UTC (EST) or ~13:00 - 20:30 UTC (EDT)
    # Simplified: 01:00 - 22:00 UTC covers both
    return 1 <= hour_utc <= 22


# ---------------------------------------------------------------------------
# HTTP persistent cache (SQLite-backed, dynamic expiry)
# ---------------------------------------------------------------------------

def _get_http_cache_expiry() -> timedelta:
    """Dynamic cache expiry: 30min during trading hours, 6h otherwise."""
    if is_trading_hours():
        return timedelta(minutes=30)
    return timedelta(hours=6)


try:
    import requests_cache

    # Create session with default expiry; actual expiry is set per-request
    http_session = requests_cache.CachedSession(
        cache_name=str(CACHE_DIR / "http_cache"),
        backend="sqlite",
        expire_after=timedelta(hours=6),  # default fallback
    )
except ImportError:
    import requests
    http_session = requests.Session()

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DataSourceError(Exception):
    """Base exception for data source errors."""
    pass


class RateLimitError(DataSourceError):
    """Raised when a vendor rate limit is hit. Triggers fallback."""
    pass


class EmptyDataError(DataSourceError):
    """Raised when a vendor returns empty data."""
    pass


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class SimpleRateLimiter:
    """Thread-safe per-vendor request throttle."""

    def __init__(self, min_interval_seconds: float):
        self.min_interval = min_interval_seconds
        self._last_called = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            elapsed = now - self._last_called
            to_sleep = self.min_interval - elapsed
            if to_sleep > 0:
                time.sleep(to_sleep)
            self._last_called = time.time()


# Pre-configured limiters for each vendor
yfinance_limiter = SimpleRateLimiter(2.0)       # 2s between calls
alpha_vantage_limiter = SimpleRateLimiter(15.0)  # 15s (free: 5/min)
tushare_limiter = SimpleRateLimiter(1.0)         # 1s between calls
akshare_limiter = SimpleRateLimiter(1.5)         # 1.5s between calls

# ---------------------------------------------------------------------------
# File cache (parquet) — with date-aware caching
# ---------------------------------------------------------------------------


def _is_historical_range(end_date: str) -> bool:
    """Check if the date range is purely historical (end_date < today).

    Historical data never changes, so it's safe to cache permanently.
    Today's data may still be updating, so skip caching.
    """
    try:
        today = date.today().strftime("%Y-%m-%d")
        # Normalize end_date format
        end_clean = end_date.replace("/", "-").strip()
        if len(end_clean) == 8 and "-" not in end_clean:
            end_clean = f"{end_clean[:4]}-{end_clean[4:6]}-{end_clean[6:]}"
        return end_clean < today
    except Exception:
        return False  # If unsure, don't cache


def _make_cache_key(source: str, symbol: str, start: str, end: str,
                    extra: str = "") -> str:
    raw = f"{source}|{symbol}|{start}|{end}|{extra}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_file_cache(source: str, symbol: str, start: str, end: str,
                    extra: str = "") -> Optional[pd.DataFrame]:
    """Load cached DataFrame from parquet file.

    Only returns cache for historical date ranges (end_date < today).
    Today's data is never served from cache.
    """
    if not _is_historical_range(end):
        return None

    key = _make_cache_key(source, symbol, start, end, extra)
    path = FILE_CACHE_DIR / f"{key}.parquet"
    if path.exists():
        try:
            return pd.read_parquet(path)
        except Exception:
            path.unlink(missing_ok=True)
    return None


def save_file_cache(df: pd.DataFrame, source: str, symbol: str,
                    start: str, end: str, extra: str = "") -> None:
    """Save DataFrame to parquet cache.

    Only caches historical data (end_date < today).
    Today's data is never written to cache.
    """
    if not _is_historical_range(end):
        return

    key = _make_cache_key(source, symbol, start, end, extra)
    path = FILE_CACHE_DIR / f"{key}.parquet"
    try:
        df.to_parquet(path, index=True)
    except Exception:
        pass  # Cache write failure is non-fatal


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential_jitter,
    )

    def resilient_call(func, *args, max_attempts=3, **kwargs):
        """Call func with retry + exponential backoff on transient errors."""
        @retry(
            retry=retry_if_exception_type((Exception,)),
            wait=wait_exponential_jitter(initial=2, max=30),
            stop=stop_after_attempt(max_attempts),
            reraise=True,
        )
        def _inner():
            return func(*args, **kwargs)
        return _inner()

except ImportError:
    def resilient_call(func, *args, max_attempts=3, **kwargs):
        """Fallback: simple retry without tenacity."""
        last_err = None
        for attempt in range(max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_err = e
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
        raise last_err


# ---------------------------------------------------------------------------
# OHLCV column normalizer
# ---------------------------------------------------------------------------


def normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard Open/High/Low/Close/Volume."""
    rename_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "adj close": "Adj Close",
        "volume": "Volume",
    }
    cols = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in rename_map:
            cols[c] = rename_map[lc]
    if cols:
        df = df.rename(columns=cols)
    return df.sort_index()
