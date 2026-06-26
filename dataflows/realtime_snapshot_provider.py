from __future__ import annotations

"""Realtime quote snapshot collection for live overnight inference.

Primary source: Tencent Finance HTTP (`qt.gtimg.cn`).
Fallback source: Tushare legacy realtime quote API (`ts.get_realtime_quotes`).

The collector normalizes vendor-specific realtime quote fields into the schema
expected by `overnight_live_provider.py`:

- ts_code
- last_price
- open
- high
- low
- pre_close
- volume
- amount
- quote_date
- quote_time
- run_ts
- source

Tushare realtime quotes expect bare 6-digit symbols (`600188`), while the rest
of this repository uses exchange-qualified Tushare symbols (`600188.SH`).  This
module preserves a mapping in both directions.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class SnapshotQuality:
    expected_count: int
    returned_count: int
    usable_count: int
    coverage_ratio: float
    min_quote_time: str | None
    max_quote_time: str | None
    stale_time_threshold: str | None = None
    freshness_ok: bool = True

    @property
    def ok(self) -> bool:
        return self.coverage_ratio >= 0.95 and self.usable_count > 0 and self.freshness_ok


def normalize_ts_code(value: str) -> str:
    s = str(value).strip().upper()
    if not s:
        return s
    if "." in s:
        code, exch = s.split(".", 1)
        return f"{code.zfill(6)}.{exch}"
    code = s.zfill(6)
    if code.startswith(("5", "6", "9")):
        return f"{code}.SH"
    return f"{code}.SZ"


def ts_code_to_realtime_symbol(ts_code: str) -> str:
    return normalize_ts_code(ts_code).split(".", 1)[0]


def realtime_symbol_to_ts_code(code: str) -> str:
    return normalize_ts_code(code)


def _chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_universe_from_feature_table(path: str | Path, trade_date: str | None = None) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Feature table not found: {p}")
    df = pd.read_csv(p, usecols=lambda c: c in {"trade_date", "ts_code"})
    if "ts_code" not in df.columns:
        raise ValueError(f"Feature table missing ts_code column: {p}")
    if trade_date and "trade_date" in df.columns:
        dates = pd.to_datetime(df["trade_date"], errors="coerce")
        cutoff = pd.to_datetime(trade_date, errors="coerce")
        df = df.loc[dates < cutoff].copy()
    if "trade_date" in df.columns and not df.empty:
        latest_date = df["trade_date"].max()
        df = df.loc[df["trade_date"] == latest_date].copy()
    codes = sorted({normalize_ts_code(x) for x in df["ts_code"].dropna().astype(str)})
    if not codes:
        raise ValueError(f"No universe symbols found in feature table: {p}")
    return codes


def standardize_tushare_realtime_quotes(raw: pd.DataFrame, requested_ts_codes: list[str]) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if "code" not in df.columns:
        raise ValueError("Tushare realtime quote result missing code column")

    symbol_to_ts = {ts_code_to_realtime_symbol(x): normalize_ts_code(x) for x in requested_ts_codes}
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["ts_code"] = df["code"].map(symbol_to_ts).fillna(df["code"].map(realtime_symbol_to_ts_code))

    rename = {
        "price": "last_price",
        "pre_close": "pre_close",
        "open": "open",
        "high": "high",
        "low": "low",
        "volume": "volume",
        "amount": "amount",
        "date": "quote_date",
        "time": "quote_time",
    }
    keep = [c for c in ["ts_code", "code", "name", *rename.keys()] if c in df.columns]
    out = df[keep].rename(columns=rename).copy()
    for col in ["last_price", "open", "high", "low", "pre_close", "volume", "amount"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["run_ts"] = datetime.now().isoformat(timespec="seconds")
    out["source"] = "tushare.get_realtime_quotes"
    out = out.drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)
    return out


def assess_snapshot_quality(snapshot: pd.DataFrame, expected_ts_codes: list[str], stale_time_threshold: str | None = None) -> SnapshotQuality:
    expected = {normalize_ts_code(x) for x in expected_ts_codes}
    usable = snapshot.copy() if snapshot is not None else pd.DataFrame()
    if not usable.empty:
        usable = usable.loc[usable.get("ts_code").astype(str).map(normalize_ts_code).isin(expected)].copy()
        if "last_price" in usable.columns:
            usable = usable.loc[pd.to_numeric(usable["last_price"], errors="coerce").gt(0)].copy()
    returned_count = int(len(snapshot)) if snapshot is not None else 0
    usable_count = int(len(usable))
    coverage = usable_count / max(len(expected), 1)
    min_time = None
    max_time = None
    if not usable.empty and "quote_time" in usable.columns:
        times = usable["quote_time"].dropna().astype(str)
        if not times.empty:
            min_time = str(times.min())
            max_time = str(times.max())
    freshness_ok = True
    if stale_time_threshold and max_time:
        freshness_ok = str(max_time) >= str(stale_time_threshold)
    elif stale_time_threshold and not max_time:
        freshness_ok = False
    return SnapshotQuality(
        expected_count=len(expected),
        returned_count=returned_count,
        usable_count=usable_count,
        coverage_ratio=float(coverage),
        min_quote_time=min_time,
        max_quote_time=max_time,
        stale_time_threshold=stale_time_threshold,
        freshness_ok=bool(freshness_ok),
    )


def fetch_tushare_realtime_snapshot(
    ts_codes: list[str],
    chunk_size: int = 300,
) -> pd.DataFrame:
    if not ts_codes:
        raise ValueError("ts_codes must not be empty")
    import tushare as ts

    normalized = [normalize_ts_code(x) for x in ts_codes]
    pairs = [(x, ts_code_to_realtime_symbol(x)) for x in normalized]
    frames: list[pd.DataFrame] = []
    for chunk_pairs in _chunked(pairs, max(1, int(chunk_size))):
        chunk_ts_codes = [ts_code for ts_code, _ in chunk_pairs]
        chunk_symbols = [symbol for _, symbol in chunk_pairs]
        raw = ts.get_realtime_quotes(chunk_symbols)
        if raw is not None and not raw.empty:
            frames.append(standardize_tushare_realtime_quotes(raw, chunk_ts_codes))
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)
    return out


def write_snapshot(snapshot: pd.DataFrame, out_path: str | Path) -> Path:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(p, index=False)
    return p


def fetch_tencent_realtime_snapshot(ts_codes: list[str], chunk_size: int = 80) -> pd.DataFrame:
    from dataflows.ashare_enrichment_provider import fetch_tencent_realtime_snapshot as _fetch

    return _fetch(ts_codes, chunk_size=chunk_size)


def fetch_realtime_snapshot_with_fallback(
    ts_codes: list[str],
    primary: str = "tencent",
    fallback: str = "tushare",
    chunk_size: int = 80,
) -> pd.DataFrame:
    from dataflows.ashare_enrichment_provider import fetch_realtime_snapshot_with_fallback as _fetch

    return _fetch(ts_codes, primary=primary, fallback=fallback, chunk_size=chunk_size)
