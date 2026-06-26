#!/usr/bin/env python3
"""Build CSI300 close-to-next-open overnight labels from Tushare daily bars.

This script intentionally avoids Tushare interfaces that the current token may
not have permission for (trade_cal, index_daily, daily_basic). It uses:
  - index_weight when available for CSI300 constituents; otherwise stock_basic
    + a deterministic top-300 proxy fallback.
  - daily for OHLC bars.

Output:
  data/overnight_labels/csi300_overnight_labels_YYYYMMDD_YYYYMMDD.csv
  data/overnight_labels/csi300_overnight_labels_audit_YYYYMMDD_YYYYMMDD.md
  data/overnight_labels/csi300_overnight_labels_audit_YYYYMMDD_YYYYMMDD.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from dataflows.tushare_provider import (
    TushareRateLimitError,
    _fmt_date,
    _get_pro,
    _parse_date,
)


DEFAULT_START = "2024-01-01"
DEFAULT_END = "2026-04-30"
DEFAULT_OUTDIR = Path("data/overnight_labels")
DEFAULT_CACHE_DIR = Path("data/tushare_daily_cache")
DEFAULT_ADJ_CACHE_DIR = Path("data/tushare_adj_factor_cache")
LOCAL_CS300_HISTORY = Path("/home/sun/.openclaw/workspace-research-main/data/constituents/cs300_history.csv")
LOCAL_STOCK_BASIC = Path("/home/sun/.openclaw/workspace-research-main/data/industry/stock_basic.csv")


def normalize_date(date: str) -> str:
    return _parse_date(date)


def get_csi300_symbols(start_date: str, end_date: str, fallback_top_n: int = 300) -> tuple[list[str], str, pd.DataFrame]:
    """Return CSI300 symbols.

    Prefer historical CSI300 constituents from index_weight. If unavailable,
    fall back to the first 300 active stocks from stock_basic so the pipeline can
    still be exercised with the current token. The audit records the source.
    """
    pro = _get_pro()
    start = normalize_date(start_date)
    end = normalize_date(end_date)

    # Try several representative dates; many Tushare accounts allow index_weight
    # only by month-end or quarter-end snapshots.
    candidate_dates = pd.date_range(start=start, end=end, freq="QE").strftime("%Y%m%d").tolist()
    candidate_dates += pd.date_range(start=start, end=end, freq="ME").strftime("%Y%m%d").tolist()[-6:]
    candidate_dates += [_fmt_date(end), _fmt_date(start)]

    frames = []
    index_errors = []
    for trade_date in dict.fromkeys(candidate_dates):
        try:
            df = pro.index_weight(index_code="000300.SH", trade_date=trade_date)
            if df is not None and not df.empty:
                df = df.copy()
                df["snapshot_trade_date"] = _parse_date(trade_date)
                frames.append(df)
                print(f"index_weight {trade_date}: {len(df)} rows")
                # One valid snapshot is enough for this first label dataset.
                break
        except Exception as exc:
            index_errors.append(f"{trade_date}: {type(exc).__name__}: {exc}")
            if "权限" in str(exc) or "访问权限" in str(exc):
                break
        time.sleep(0.12)

    if frames:
        constituents = pd.concat(frames, ignore_index=True)
        if "con_code" in constituents.columns:
            codes = constituents["con_code"].dropna().astype(str).str.upper().drop_duplicates().sort_values().tolist()
        else:
            codes = constituents.iloc[:, 0].dropna().astype(str).str.upper().drop_duplicates().sort_values().tolist()
        return codes, "tushare.index_weight:000300.SH", constituents

    if LOCAL_CS300_HISTORY.exists():
        local = pd.read_csv(LOCAL_CS300_HISTORY)
        local = local.copy()
        local["trade_date_norm"] = local["trade_date"].astype(str).map(_parse_date)
        local = local[(local["trade_date_norm"] >= start) & (local["trade_date_norm"] <= end)]
        if local.empty:
            all_local = pd.read_csv(LOCAL_CS300_HISTORY)
            all_local = all_local.copy()
            all_local["trade_date_norm"] = all_local["trade_date"].astype(str).map(_parse_date)
            latest_date = all_local["trade_date_norm"].max()
            local = all_local[all_local["trade_date_norm"] == latest_date].copy()
        codes = local["con_code"].dropna().astype(str).str.upper().drop_duplicates().sort_values().tolist()
        local["fallback_reason"] = "; ".join(index_errors[:3]) if index_errors else "online index_weight returned empty"
        return codes, f"local_cs300_history:{LOCAL_CS300_HISTORY}", local

    print("index_weight unavailable; falling back to stock_basic top-300 proxy")
    if LOCAL_STOCK_BASIC.exists():
        stock_basic = pd.read_csv(LOCAL_STOCK_BASIC)
    else:
        stock_basic = pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name,area,industry,list_date")
    if stock_basic is None or stock_basic.empty:
        raise RuntimeError("stock_basic returned no rows; cannot build fallback stock universe")
    stock_basic = stock_basic.copy()
    stock_basic["list_date"] = stock_basic["list_date"].astype(str)
    stock_basic = stock_basic.sort_values(["list_date", "ts_code"]).head(fallback_top_n).reset_index(drop=True)
    codes = stock_basic["ts_code"].astype(str).str.upper().tolist()
    stock_basic["fallback_reason"] = "; ".join(index_errors[:3]) if index_errors else "index_weight returned empty"
    return codes, f"stock_basic_top_{fallback_top_n}_fallback_not_true_csi300", stock_basic


def _normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df["trade_date"] = df["trade_date"].map(_parse_date)
    keep = [c for c in ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"] if c in df.columns]
    return df[keep].sort_values("trade_date").reset_index(drop=True)


def fetch_daily(
    ts_code: str,
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    retries: int = 6,
    sleep_s: float = 1.4,
) -> tuple[pd.DataFrame, str | None]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{ts_code}_{_fmt_date(start_date)}_{_fmt_date(end_date)}.csv"
        if use_cache and cache_path.exists():
            try:
                cached = pd.read_csv(cache_path)
                cached = _normalize_daily_df(cached)
                if not cached.empty:
                    return cached, None
            except Exception:
                pass

    pro = _get_pro()
    last_msg = None
    for attempt in range(1, retries + 1):
        try:
            df = pro.daily(ts_code=ts_code, start_date=_fmt_date(start_date), end_date=_fmt_date(end_date))
            if df is None or df.empty:
                return pd.DataFrame(), "empty_daily"
            df = _normalize_daily_df(df)
            if cache_path is not None:
                df.to_csv(cache_path, index=False)
            return df, None
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            last_msg = msg
            if "频率超限" in str(exc) or "每分钟" in str(exc):
                wait = max(65.0, sleep_s * attempt)
                print(f"    rate limit hit for {ts_code}; sleeping {wait:.1f}s before retry {attempt}/{retries}", flush=True)
                time.sleep(wait)
                continue
            if attempt == retries:
                return pd.DataFrame(), msg
            time.sleep(sleep_s * attempt)
    return pd.DataFrame(), last_msg or "unknown_fetch_failure"


def fetch_adj_factor(
    ts_code: str,
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    use_cache: bool = True,
    retries: int = 3,
    sleep_s: float = 1.4,
) -> tuple[pd.DataFrame, str | None]:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{ts_code}_{_fmt_date(start_date)}_{_fmt_date(end_date)}.csv"
        if use_cache and cache_path.exists():
            try:
                cached = pd.read_csv(cache_path)
                if not cached.empty:
                    cached["trade_date"] = cached["trade_date"].map(_parse_date)
                    return cached.sort_values("trade_date").reset_index(drop=True), None
            except Exception:
                pass

    pro = _get_pro()
    last_msg = None
    for attempt in range(1, retries + 1):
        try:
            df = pro.adj_factor(ts_code=ts_code, start_date=_fmt_date(start_date), end_date=_fmt_date(end_date))
            if df is None or df.empty:
                return pd.DataFrame(), "empty_adj_factor"
            df = df.copy()
            df["trade_date"] = df["trade_date"].map(_parse_date)
            df = df[[c for c in ["ts_code", "trade_date", "adj_factor"] if c in df.columns]].sort_values("trade_date").reset_index(drop=True)
            if cache_path is not None:
                df.to_csv(cache_path, index=False)
            return df, None
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            last_msg = msg
            if "频率超限" in str(exc) or "每分钟" in str(exc):
                wait = max(65.0, sleep_s * attempt)
                print(f"    adj_factor rate limit hit for {ts_code}; sleeping {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            if "权限" in str(exc) or "访问权限" in str(exc):
                return pd.DataFrame(), msg
            time.sleep(sleep_s * attempt)
    return pd.DataFrame(), last_msg or "unknown_adj_factor_failure"


def apply_forward_adjustment(bars: pd.DataFrame, adj: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if bars.empty or adj.empty or "adj_factor" not in adj.columns:
        return bars, "raw_unadjusted"
    merged = bars.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
    merged["adj_factor"] = pd.to_numeric(merged["adj_factor"], errors="coerce").ffill().bfill()
    if merged["adj_factor"].isna().all():
        return bars, "raw_unadjusted_adj_factor_missing"
    latest = float(merged["adj_factor"].iloc[-1])
    if latest == 0:
        return bars, "raw_unadjusted_adj_factor_zero"
    ratio = merged["adj_factor"] / latest
    out = merged.copy()
    for col in ["open", "high", "low", "close", "pre_close"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * ratio
    return out.drop(columns=["adj_factor"]), "tushare.daily+adj_factor_forward"


def build_labels_from_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty or len(bars) < 2:
        return pd.DataFrame()
    df = bars.sort_values("trade_date").reset_index(drop=True).copy()
    df["next_trade_date"] = df["trade_date"].shift(-1)
    df["next_open"] = df["open"].shift(-1)
    df = df.iloc[:-1].copy()
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["next_open"] = pd.to_numeric(df["next_open"], errors="coerce")
    df["overnight_return_open"] = df["next_open"] / df["close"] - 1.0
    df["gap_days"] = (pd.to_datetime(df["next_trade_date"]) - pd.to_datetime(df["trade_date"])).dt.days
    df["source"] = df["source"].iloc[0] if "source" in df.columns and df["source"].notna().any() else "tushare.daily"
    df["is_valid"] = df["close"].notna() & df["next_open"].notna() & (df["close"] > 0)
    return df[[
        "ts_code", "trade_date", "close", "next_trade_date", "next_open",
        "overnight_return_open", "gap_days", "source", "is_valid"
    ]]


def audit_labels(labels: pd.DataFrame, failures: list[dict], symbols: list[str], universe_source: str, start_date: str, end_date: str) -> dict:
    valid = labels[labels["is_valid"] == True].copy() if not labels.empty else labels
    ret = pd.to_numeric(valid["overnight_return_open"], errors="coerce") if not valid.empty else pd.Series(dtype=float)
    by_year = valid.assign(year=valid["trade_date"].str.slice(0, 4)).groupby("year").size().to_dict() if not valid.empty else {}
    by_symbol = valid.groupby("ts_code").size() if not valid.empty else pd.Series(dtype=int)
    extreme = valid[ret.abs() > 0.20].copy() if not valid.empty else pd.DataFrame()
    audit = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "start_date": start_date,
        "end_date": end_date,
        "universe_source": universe_source,
        "n_symbols_requested": len(symbols),
        "n_symbols_with_labels": int(valid["ts_code"].nunique()) if not valid.empty else 0,
        "n_symbols_failed": len(failures),
        "n_label_rows": int(len(labels)),
        "n_valid_rows": int(len(valid)),
        "n_invalid_rows": int((labels["is_valid"] != True).sum()) if not labels.empty and "is_valid" in labels else 0,
        "return_summary": {
            "mean": float(ret.mean()) if len(ret) else None,
            "std": float(ret.std()) if len(ret) else None,
            "min": float(ret.min()) if len(ret) else None,
            "p01": float(ret.quantile(0.01)) if len(ret) else None,
            "p05": float(ret.quantile(0.05)) if len(ret) else None,
            "median": float(ret.median()) if len(ret) else None,
            "p95": float(ret.quantile(0.95)) if len(ret) else None,
            "p99": float(ret.quantile(0.99)) if len(ret) else None,
            "max": float(ret.max()) if len(ret) else None,
        },
        "gap_days_summary": valid["gap_days"].describe().to_dict() if not valid.empty else {},
        "rows_by_year": {str(k): int(v) for k, v in by_year.items()},
        "labels_per_symbol": {
            "min": int(by_symbol.min()) if len(by_symbol) else 0,
            "median": float(by_symbol.median()) if len(by_symbol) else 0,
            "max": int(by_symbol.max()) if len(by_symbol) else 0,
        },
        "n_extreme_abs_return_gt_20pct": int(len(extreme)),
        "failures_head": failures[:20],
        "extreme_head": extreme.head(20).to_dict(orient="records") if not extreme.empty else [],
    }
    return audit


def write_audit_md(audit: dict, path: Path, labels_path: Path, universe_path: Path) -> None:
    rs = audit["return_summary"]
    lines = [
        "# CSI300 Overnight Labels Quality Audit",
        "",
        f"- Created at: `{audit['created_at']}`",
        f"- Date range: `{audit['start_date']}` to `{audit['end_date']}`",
        f"- Universe source: `{audit['universe_source']}`",
        f"- Labels CSV: `{labels_path}`",
        f"- Universe file: `{universe_path}`",
        "",
        "## Coverage",
        "",
        f"- Requested symbols: `{audit['n_symbols_requested']}`",
        f"- Symbols with labels: `{audit['n_symbols_with_labels']}`",
        f"- Failed symbols: `{audit['n_symbols_failed']}`",
        f"- Label rows: `{audit['n_label_rows']}`",
        f"- Valid rows: `{audit['n_valid_rows']}`",
        f"- Invalid rows: `{audit['n_invalid_rows']}`",
        "",
        "## Return Distribution",
        "",
        f"- Mean: `{rs['mean']}`",
        f"- Std: `{rs['std']}`",
        f"- Min / P01 / P05: `{rs['min']}` / `{rs['p01']}` / `{rs['p05']}`",
        f"- Median: `{rs['median']}`",
        f"- P95 / P99 / Max: `{rs['p95']}` / `{rs['p99']}` / `{rs['max']}`",
        f"- Extreme abs return > 20% rows: `{audit['n_extreme_abs_return_gt_20pct']}`",
        "",
        "## Rows by Year",
        "",
    ]
    for year, n in audit["rows_by_year"].items():
        lines.append(f"- `{year}`: `{n}`")
    lines += [
        "",
        "## Labels Per Symbol",
        "",
        f"- Min: `{audit['labels_per_symbol']['min']}`",
        f"- Median: `{audit['labels_per_symbol']['median']}`",
        f"- Max: `{audit['labels_per_symbol']['max']}`",
        "",
        "## Failures Head",
        "",
    ]
    if audit["failures_head"]:
        for item in audit["failures_head"]:
            lines.append(f"- `{item.get('ts_code')}`: `{item.get('error')}`")
    else:
        lines.append("- None")
    lines += ["", "## Extreme Returns Head", ""]
    if audit["extreme_head"]:
        for item in audit["extreme_head"]:
            lines.append(f"- `{item.get('ts_code')}` `{item.get('trade_date')}` return=`{item.get('overnight_return_open')}`")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default=DEFAULT_START)
    parser.add_argument("--end-date", default=DEFAULT_END)
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--no-cache", action="store_true", help="ignore and overwrite cached daily bars")
    parser.add_argument("--adjust", choices=["none", "forward"], default="forward", help="price adjustment mode; forward uses Tushare adj_factor when permitted")
    parser.add_argument("--adj-cache-dir", default=str(DEFAULT_ADJ_CACHE_DIR))
    parser.add_argument("--sleep", type=float, default=0.18, help="sleep seconds between symbols")
    parser.add_argument("--limit", type=int, default=0, help="debug limit; 0 means all")
    args = parser.parse_args()

    start = normalize_date(args.start_date)
    end = normalize_date(args.end_date)
    outdir = Path(args.outdir)
    cache_dir = Path(args.cache_dir)
    adj_cache_dir = Path(args.adj_cache_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"{start.replace('-', '')}_{end.replace('-', '')}"

    symbols, universe_source, universe_df = get_csi300_symbols(start, end)
    if args.limit:
        symbols = symbols[: args.limit]
        universe_source += f"_limit_{args.limit}"

    universe_path = outdir / f"csi300_universe_{tag}.csv"
    universe_df.to_csv(universe_path, index=False)

    all_labels = []
    failures = []
    for idx, ts_code in enumerate(symbols, start=1):
        bars, error = fetch_daily(ts_code, start, end, cache_dir=cache_dir, use_cache=not args.no_cache, sleep_s=args.sleep)
        if error:
            failures.append({"ts_code": ts_code, "error": error})
            print(f"[{idx}/{len(symbols)}] {ts_code}: FAIL {error}")
        else:
            source = "tushare.daily"
            if args.adjust == "forward":
                adj, adj_error = fetch_adj_factor(ts_code, start, end, cache_dir=adj_cache_dir, use_cache=not args.no_cache, sleep_s=args.sleep)
                if adj_error:
                    print(f"[{idx}/{len(symbols)}] {ts_code}: adj_factor unavailable ({adj_error}); using raw daily")
                bars, source = apply_forward_adjustment(bars, adj)
            bars = bars.copy()
            bars["source"] = source
            labels = build_labels_from_bars(bars)
            if labels.empty:
                failures.append({"ts_code": ts_code, "error": "insufficient_bars_for_labels"})
                print(f"[{idx}/{len(symbols)}] {ts_code}: FAIL insufficient bars")
            else:
                all_labels.append(labels)
                print(f"[{idx}/{len(symbols)}] {ts_code}: {len(labels)} labels")
        time.sleep(args.sleep)

    labels = pd.concat(all_labels, ignore_index=True) if all_labels else pd.DataFrame()
    labels_path = outdir / f"csi300_overnight_labels_{tag}.csv"
    labels.to_csv(labels_path, index=False)

    audit = audit_labels(labels, failures, symbols, universe_source, start, end)
    audit_json = outdir / f"csi300_overnight_labels_audit_{tag}.json"
    audit_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_md = outdir / f"csi300_overnight_labels_audit_{tag}.md"
    write_audit_md(audit, audit_md, labels_path, universe_path)

    print("\nDONE")
    print(f"labels: {labels_path}")
    print(f"audit_md: {audit_md}")
    print(f"audit_json: {audit_json}")
    print(json.dumps({k: audit[k] for k in ["universe_source", "n_symbols_requested", "n_symbols_with_labels", "n_symbols_failed", "n_valid_rows", "n_extreme_abs_return_gt_20pct"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
