"""
Multi-source realtime snapshot fallback chain — Vibe-Trading pattern.

Ordered degradation: tencent → eastmoney → baostock → akshare → tushare.
Principle: never-banned public FIRST, throttled NEXT, key-gated LAST.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

import pandas as pd


FALLBACK_CHAIN: list[tuple[str, Callable]] = []


def _register(name: str):
    """Decorator: register a fetcher into FALLBACK_CHAIN."""
    def decorator(fn: Callable) -> Callable:
        FALLBACK_CHAIN.append((name, fn))
        return fn
    return decorator


from dataflows.realtime_snapshot_provider import (
    normalize_ts_code,
    fetch_tushare_realtime_snapshot,
)
from dataflows.ashare_enrichment_provider import fetch_tencent_realtime_snapshot

# Register tencent as primary (imported, not decorated)
_register("tencent")(fetch_tencent_realtime_snapshot)


# ── Eastmoney fetcher (via akshare stock_zh_a_spot_em) ──────────
@_register("eastmoney")
def _fetch_eastmoney(ts_codes: list[str]) -> pd.DataFrame:
    try:
        import akshare as ak
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return pd.DataFrame()
        code_col = "代码" if "代码" in df.columns else None
        if not code_col:
            return pd.DataFrame()
        df["ts_code"] = df[code_col].astype(str).str.zfill(6).map(normalize_ts_code)
        df = df[df["ts_code"].isin(ts_codes)].copy()
        rename = {
            "最新价": "last_price", "今开": "open", "最高": "high",
            "最低": "low", "昨收": "pre_close", "成交量": "volume", "成交额": "amount",
        }
        out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
        out = out[["ts_code"] + [c for c in rename.values() if c in out.columns]].copy()
        for col in rename.values():
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce")
        out["source"] = "eastmoney"
        return out.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


# ── Baostock fetcher ─────────────────────────────────────────────
@_register("baostock")
def _fetch_baostock(ts_codes: list[str]) -> pd.DataFrame:
    try:
        import baostock as bs
        lg = bs.login()
        if lg.error_code != "0":
            return pd.DataFrame()
        rows = []
        for code in ts_codes:
            bare = code.split(".", 1)[0] if "." in code else code
            market = "sh" if bare.startswith(("6", "9", "5")) else "sz"
            rs = bs.query_history_k_data_plus(f"{market}.{bare}",
                "date,open,high,low,close,preclose,volume,amount",
                frequency="d", adjustflag="3")
            data = []
            while (rs.error_code == "0") and rs.next():
                data.append(rs.get_row_data())
            if data:
                latest = data[-1]
                rows.append({
                    "ts_code": normalize_ts_code(bare),
                    "last_price": float(latest[3]),
                    "open": float(latest[1]), "high": float(latest[2]),
                    "low": float(latest[3]), "pre_close": float(latest[4]),
                    "volume": float(latest[5]), "amount": float(latest[6]),
                    "source": "baostock",
                })
            time.sleep(0.02)
        bs.logout()
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── AkShare fallback (latest daily bar as pseudo-realtime) ──────
@_register("akshare")
def _fetch_akshare_fallback(ts_codes: list[str]) -> pd.DataFrame:
    try:
        import akshare as ak
        rows = []
        for code in ts_codes:
            bare = code.split(".", 1)[0] if "." in code else code
            try:
                df = ak.stock_zh_a_hist(symbol=bare, period="daily", adjust="qfq")
                if df is not None and not df.empty:
                    latest = df.iloc[-1]
                    rows.append({
                        "ts_code": normalize_ts_code(bare),
                        "last_price": float(latest["收盘"]),
                        "open": float(latest["开盘"]), "high": float(latest["最高"]),
                        "low": float(latest["最低"]),
                        "pre_close": float(latest.get("昨收", 0)),
                        "volume": float(latest["成交量"]),
                        "amount": float(latest["成交额"]),
                        "source": "akshare",
                    })
            except Exception:
                continue
            time.sleep(0.05)
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── Tushare wrapper (key-gated, last resort) ──────────────────────
@_register("tushare")
def _fetch_tushare(ts_codes: list[str]) -> pd.DataFrame:
    """Last-resort fetcher via tushare (requires API token)."""
    try:
        df = fetch_tushare_realtime_snapshot(ts_codes, chunk_size=300)
        if df is not None and not df.empty:
            df["source"] = "tushare"
            return df
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


# ── Orchestration: fetch with ordered degradation ────────────────
def fetch_multi_source_snapshot(
    ts_codes: list[str],
    require_source: str | None = None,
    skip: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch realtime snapshot via ordered fallback chain.

    Iterates FALLBACK_CHAIN in registration order. Each source gets
    the full ts_codes list. Returns the first non-empty result.

    Returns DataFrame with ts_code + market-data columns + 'source'.
    """
    skip = skip or []
    target = set(ts_codes)
    if not target:
        return pd.DataFrame()

    if require_source:
        chain = [(n, f) for n, f in FALLBACK_CHAIN if n == require_source]
    else:
        chain = [(n, f) for n, f in FALLBACK_CHAIN if n not in skip]

    for name, fetcher in chain:
        try:
            missing = sorted(target)
            df = fetcher(missing)
            if df is not None and not df.empty:
                df["chain_source"] = name
                return df.reset_index(drop=True)
        except Exception:
            continue

    return pd.DataFrame()
