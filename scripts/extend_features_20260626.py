#!/usr/bin/env python3
"""Extend overnight feature table to June 26 — fast parallel fetch."""

import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np

FEATURE_TABLE = Path("data/overnight_mvp/features/overnight_features_20260101_20260430.csv")
OUT_FEATURE = Path("data/overnight_mvp/features/overnight_features_ext_20260626.csv")
OUT_SNAPSHOT = Path("data/snapshots/live_snapshot_20260626.csv")
LOOKBACK_DAYS = 80
MAX_WORKERS = 20
UA = "Mozilla/5.0"


def fetch_one_stock(ts_code: str) -> tuple[str, list[dict] | None]:
    """Fetch daily K-line for one stock. Returns (ts_code, rows or None)."""
    if ts_code.endswith(".SH"):
        key = f"sh{ts_code[:6]}"
    else:
        key = f"sz{ts_code[:6]}"
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={key},day,,,{LOOKBACK_DAYS},qfq"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://gu.qq.com/"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return ts_code, None
    stock_data_section = data.get("data", {})
    if not isinstance(stock_data_section, dict):
        return ts_code, None
    stock_data = stock_data_section.get(key)
    if not stock_data or "qfqday" not in stock_data:
        return ts_code, None
    raw = stock_data["qfqday"]
    if not raw or len(raw) < 5:
        return ts_code, None
    rows = []
    for item in raw:
        if not isinstance(item, list) or len(item) < 6:
            continue
        rows.append({
            "date": item[0],
            "open": float(item[1]),
            "close": float(item[2]),
            "high": float(item[3]),
            "low": float(item[4]),
            "volume": float(item[5]),
        })
    return ts_code, rows if rows else None


def compute_features(klines: list[dict], ts_code: str) -> dict | None:
    df = pd.DataFrame(klines)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if len(df) < 12:
        return None

    latest = df.iloc[-1]
    close = latest["close"]
    trade_date = latest["date"].strftime("%Y-%m-%d")

    closes = df["close"]
    highs = df["high"]
    lows = df["low"]

    def _ret_n(n):
        if len(closes) < n + 1:
            return np.nan
        return closes.iloc[-1] / closes.iloc[-n - 1] - 1

    ret_close_1d = _ret_n(1)
    ret_close_3d = _ret_n(3)
    ret_close_5d = _ret_n(5)

    ma5 = closes.iloc[-5:].mean() if len(closes) >= 5 else np.nan
    close_ma5_ratio = close / ma5 - 1 if ma5 and ma5 > 0 else np.nan

    ma10 = closes.iloc[-10:].mean() if len(closes) >= 10 else np.nan
    close_ma10_ratio = close / ma10 - 1 if ma10 and ma10 > 0 else np.nan

    if len(highs) >= 5 and len(lows) >= 5:
        h5, l5 = highs.iloc[-5:].max(), lows.iloc[-5:].min()
        rng = h5 - l5
        close_range_pos_5d = (close - l5) / rng if rng > 0 else 0.5
    else:
        close_range_pos_5d = np.nan

    if len(highs) >= 10:
        h10 = highs.iloc[-10:].max()
        close_drawdown_10d = close / h10 - 1 if h10 > 0 else np.nan
    else:
        close_drawdown_10d = np.nan

    if len(closes) >= 6:
        close_vol_5d = closes.pct_change().iloc[-5:].std()
    else:
        close_vol_5d = np.nan

    opens = df["open"]
    overnight_rets = []
    for idx in range(1, len(df)):
        prev_close = closes.iloc[idx - 1]
        curr_open = opens.iloc[idx]
        if prev_close > 0:
            overnight_rets.append(curr_open / prev_close - 1)

    overnight_prev_1d = overnight_rets[-1] if len(overnight_rets) >= 1 else np.nan
    overnight_prev_3d_mean = np.mean(overnight_rets[-3:]) if len(overnight_rets) >= 3 else np.nan
    overnight_prev_5d_mean = np.mean(overnight_rets[-5:]) if len(overnight_rets) >= 5 else np.nan
    overnight_prev_5d_std = np.std(overnight_rets[-5:]) if len(overnight_rets) >= 5 else np.nan
    positive_rate_5d = sum(1 for r in overnight_rets[-5:] if r > 0) / 5 if len(overnight_rets) >= 5 else np.nan

    row = {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "close": close,
        "open": latest["open"],
        "high": latest["high"],
        "low": latest["low"],
        "ret_close_1d": ret_close_1d,
        "ret_close_3d": ret_close_3d,
        "ret_close_5d": ret_close_5d,
        "close_ma5_ratio": close_ma5_ratio,
        "close_ma10_ratio": close_ma10_ratio,
        "close_range_pos_5d": close_range_pos_5d,
        "close_drawdown_10d": close_drawdown_10d,
        "close_vol_5d": close_vol_5d,
        "overnight_prev_1d": overnight_prev_1d,
        "overnight_prev_3d_mean": overnight_prev_3d_mean,
        "overnight_prev_5d_mean": overnight_prev_5d_mean,
        "overnight_prev_5d_std": overnight_prev_5d_std,
        "overnight_positive_rate_5d": positive_rate_5d,
    }
    return row


def main():
    print("Loading feature table...", flush=True)
    ft = pd.read_csv(FEATURE_TABLE)
    stocks = sorted(ft["ts_code"].unique().tolist())
    print(f"  {len(stocks)} unique stocks", flush=True)

    # Get metadata
    meta = ft.groupby("ts_code")[["name", "industry", "market"]].last().reset_index()
    meta_dict = {}
    for _, r in meta.iterrows():
        meta_dict[r["ts_code"]] = {
            "name": r.get("name", r["ts_code"]),
            "industry": r.get("industry", ""),
            "market": r.get("market", "主板"),
        }

    # Parallel fetch
    print(f"Fetching K-line for {len(stocks)} stocks (parallel, {MAX_WORKERS} workers)...", flush=True)
    kline_data: dict[str, list[dict]] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one_stock, code): code for code in stocks}
        for i, fut in enumerate(as_completed(futures), 1):
            code, rows = fut.result()
            if rows:
                kline_data[code] = rows
            if i % 50 == 0:
                print(f"  {i}/{len(stocks)} done ({len(kline_data)} with data)", flush=True)
    elapsed = time.time() - t0
    print(f"  Got data for {len(kline_data)}/{len(stocks)} stocks in {elapsed:.1f}s", flush=True)

    # Compute features
    print("Computing features...", flush=True)
    rows = []
    for ts_code, klines in kline_data.items():
        feat = compute_features(klines, ts_code)
        if feat is None:
            continue
        meta_info = meta_dict.get(ts_code, {})
        feat["name"] = meta_info.get("name", ts_code)
        feat["industry"] = meta_info.get("industry", "")
        feat["market"] = meta_info.get("market", "主板")
        rows.append(feat)

    df_feat = pd.DataFrame(rows)
    print(f"  Computed features for {len(df_feat)} stocks", flush=True)

    if df_feat.empty:
        print("ERROR: No features computed!", flush=True)
        sys.exit(1)

    date_counts = df_feat["trade_date"].value_counts()
    predominant_date = date_counts.index[0]
    print(f"  Trade dates: top={predominant_date} ({date_counts.iloc[0]}), next={date_counts.index[1] if len(date_counts)>1 else 'N/A'}", flush=True)

    # Write feature table
    print("Writing feature table...", flush=True)
    df_feat.to_csv(OUT_FEATURE, index=False)
    print(f"  Wrote {OUT_FEATURE} ({len(df_feat)} rows)", flush=True)

    # Write snapshot
    print("Building snapshot CSV...", flush=True)
    snap_rows = []
    for _, r in df_feat.iterrows():
        code = r["ts_code"]
        snap = {
            "ts_code": code,
            "last_price": r["close"],
            "open": r.get("open", r["close"]),
            "pre_close": r["close"],
            "high": r.get("high", r["close"]),
            "low": r.get("low", r["close"]),
        }
        snap_rows.append(snap)

    df_snap = pd.DataFrame(snap_rows)
    df_snap.to_csv(OUT_SNAPSHOT, index=False)
    print(f"  Wrote {OUT_SNAPSHOT} ({len(df_snap)} rows)", flush=True)

    print(f"\nDone. trade_date: {predominant_date} ({date_counts.iloc[0]} stocks)")


if __name__ == "__main__":
    main()
