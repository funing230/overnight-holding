#!/usr/bin/env python3
"""Build unified overnight feature table + first-pass Top-N baseline input.

This script merges the existing clean close-to-next-open label table with
feature sources that are useful for an overnight-holding strategy.

Default safe mode:
- stock_basic is loaded from cache, or fetched once if the cache is missing
- daily_basic / moneyflow / stk_factor are loaded from existing cache CSVs only
- no remote daily_basic / moneyflow / stk_factor API calls are made unless
  --fetch-tushare-factors is passed explicitly
- optional same-day 14:30-15:00 minute-window features can be merged separately

Optional remote Tushare factor sources:
- daily_basic
- moneyflow
- stk_factor

Outputs:
  data/overnight_mvp/features/overnight_features_<start>_<end>.csv
  data/overnight_mvp/backtest_inputs/topn_baseline_input_<start>_<end>.csv
  data/overnight_mvp/audit/overnight_feature_build_<start>_<end>.md

Notes:
- It expects the label table to exist already.
- It prefers current repo .env / environment TUSHARE settings.
- When an upstream call fails for a symbol, the script keeps going and records
  missingness in the audit so the pipeline stays inspectable.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

from dataflows.overnight_minute_cache import (
    DEFAULT_MINUTE_CACHE_DIR,
    fetch_minute_window_frame,
    summarize_minute_features,
)
from dataflows.tushare_provider import _fmt_date, _get_pro, _parse_date


DEFAULT_LABELS = Path("data/overnight_labels/csi300_overnight_labels_clean_20240101_20260430.csv")
DEFAULT_OUT_ROOT = Path("data/overnight_mvp")
DEFAULT_TOP_N = 10
DEFAULT_LOOKBACK_START = "2026-04-01"
DEFAULT_LOOKBACK_END = "2026-04-30"


def _normalize_date(value: str) -> str:
    return _parse_date(str(value).strip())


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_labels(path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    df = pd.read_csv(path)
    df["trade_date"] = df["trade_date"].map(_normalize_date)
    df["next_trade_date"] = df["next_trade_date"].map(_normalize_date)
    mask = (df["trade_date"] >= start_date) & (df["trade_date"] <= end_date)
    df = df.loc[mask].copy()
    if "is_trainable" in df.columns:
        trainable = df["is_trainable"].astype(str).str.lower() == "true"
        df = df.loc[trainable].copy()
    if df.empty:
        raise ValueError(f"No trainable labels found for {start_date}..{end_date} in {path}")
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def _cache_path(cache_dir: Path, prefix: str, ts_code: str, start_date: str, end_date: str) -> Path:
    safe = ts_code.replace("/", "_")
    return cache_dir / f"{prefix}_{safe}_{_fmt_date(start_date)}_{_fmt_date(end_date)}.csv"


def fetch_stock_basic(cache_dir: Path, force_refresh: bool = False) -> pd.DataFrame:
    _ensure_dir(cache_dir)
    cache_path = cache_dir / "stock_basic_all.csv"
    if cache_path.exists() and not force_refresh:
        df = pd.read_csv(cache_path)
        if not df.empty:
            return df

    try:
        pro = _get_pro()
        df = pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date",
        )
    except Exception as exc:
        if cache_path.exists():
            print(f"  stock_basic refresh failed; using cached stock_basic_all.csv: {type(exc).__name__}: {exc}", flush=True)
            df = pd.read_csv(cache_path)
            if "list_date" in df.columns:
                df["list_date"] = df["list_date"].map(_normalize_date)
            return df
        print(f"  stock_basic fetch failed and no cache exists: {type(exc).__name__}: {exc}", flush=True)
        return pd.DataFrame(columns=["ts_code", "symbol", "name", "area", "industry", "market", "list_date"])

    if df is None or df.empty:
        if cache_path.exists():
            print("  stock_basic refresh returned empty; using cached stock_basic_all.csv", flush=True)
            df = pd.read_csv(cache_path)
            if "list_date" in df.columns:
                df["list_date"] = df["list_date"].map(_normalize_date)
            return df
        return pd.DataFrame(columns=["ts_code", "symbol", "name", "area", "industry", "market", "list_date"])
    df = df.copy()
    if "list_date" in df.columns:
        df["list_date"] = df["list_date"].map(_normalize_date)
    df.to_csv(cache_path, index=False)
    return df


def _is_rate_limit_error(message: str) -> bool:
    msg = str(message)
    return any(kw in msg for kw in ["频率超限", "每分钟", "每小时", "rate limit", "2次/秒", "2次/分钟", "2次/天"])


def fetch_symbol_frame(
    api_name: str,
    ts_code: str,
    start_date: str,
    end_date: str,
    cache_dir: Path,
    force_refresh: bool = False,
    max_retries: int = 4,
    allow_remote: bool = False,
) -> tuple[pd.DataFrame, str | None]:
    _ensure_dir(cache_dir)
    cache_path = _cache_path(cache_dir, api_name, ts_code, start_date, end_date)
    if cache_path.exists() and not force_refresh:
        try:
            df = pd.read_csv(cache_path)
            if "trade_date" in df.columns:
                df["trade_date"] = df["trade_date"].map(_normalize_date)
            return df, None
        except Exception:
            pass

    if not allow_remote:
        return pd.DataFrame(), "cache_miss_remote_disabled"

    pro = _get_pro()
    last_err: str | None = None
    for attempt in range(max_retries + 1):
        try:
            if api_name == "daily_basic":
                df = pro.daily_basic(
                    ts_code=ts_code,
                    start_date=_fmt_date(start_date),
                    end_date=_fmt_date(end_date),
                    fields="ts_code,trade_date,turnover_rate,volume_ratio,pe,pb,total_mv,circ_mv",
                )
            elif api_name == "moneyflow":
                df = pro.moneyflow(
                    ts_code=ts_code,
                    start_date=_fmt_date(start_date),
                    end_date=_fmt_date(end_date),
                )
            elif api_name == "stk_factor":
                df = pro.stk_factor(
                    ts_code=ts_code,
                    start_date=_fmt_date(start_date),
                    end_date=_fmt_date(end_date),
                )
            else:
                raise ValueError(f"Unsupported api_name={api_name}")
            break
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if _is_rate_limit_error(exc) and attempt < max_retries:
                sleep_s = 2.5 * (attempt + 1)
                print(f"  rate-limited on {api_name} {ts_code}; retry {attempt + 1}/{max_retries} after {sleep_s:.1f}s", flush=True)
                time.sleep(sleep_s)
                continue
            return pd.DataFrame(), last_err

    if df is None or df.empty:
        return pd.DataFrame(), "empty"

    df = df.copy()
    if "trade_date" in df.columns:
        df["trade_date"] = df["trade_date"].map(_normalize_date)
    df.to_csv(cache_path, index=False)
    return df, None




def merge_feature_frames(
    labels: pd.DataFrame,
    stock_basic: pd.DataFrame,
    daily_basic: pd.DataFrame,
    moneyflow: pd.DataFrame,
    stk_factor: pd.DataFrame,
    minute_features: pd.DataFrame,
) -> pd.DataFrame:
    df = labels.copy()

    if not stock_basic.empty:
        keep = [c for c in ["ts_code", "name", "area", "industry", "market", "list_date"] if c in stock_basic.columns]
        df = df.merge(stock_basic[keep].drop_duplicates(subset=["ts_code"]), on="ts_code", how="left")

    if not daily_basic.empty:
        keep = [c for c in ["ts_code", "trade_date", "turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv"] if c in daily_basic.columns]
        df = df.merge(daily_basic[keep], on=["ts_code", "trade_date"], how="left")

    if not moneyflow.empty:
        keep = [c for c in [
            "ts_code", "trade_date",
            "buy_elg_amount", "buy_lg_amount", "buy_md_amount", "buy_sm_amount",
            "sell_elg_amount", "sell_lg_amount", "sell_md_amount", "sell_sm_amount",
            "net_mf_amount", "net_mf_vol",
        ] if c in moneyflow.columns]
        df = df.merge(moneyflow[keep], on=["ts_code", "trade_date"], how="left")

    if not stk_factor.empty:
        keep = [c for c in [
            "ts_code", "trade_date",
            "open", "high", "low", "close", "pre_close", "change", "pct_change", "vol", "amount", "adj_factor",
            "macd_dif", "macd_dea", "macd",
            "kdj_k", "kdj_d", "kdj_j",
            "rsi_6", "rsi_12", "rsi_24",
            "boll_upper", "boll_mid", "boll_lower", "cci",
        ] if c in stk_factor.columns]
        rename_map = {"close": "close_stk_factor", "open": "open_t", "high": "high_t", "low": "low_t", "vol": "vol_t", "amount": "amount_t"}
        df = df.merge(stk_factor[keep].rename(columns=rename_map), on=["ts_code", "trade_date"], how="left")

    if not minute_features.empty:
        df = df.merge(minute_features, on=["ts_code", "trade_date"], how="left")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["trade_date"] = pd.to_datetime(out.get("trade_date"), errors="coerce")
    out = out.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

    close_series = pd.to_numeric(out.get("close"), errors="coerce")
    next_open_series = pd.to_numeric(out.get("next_open"), errors="coerce")

    by_symbol = out.groupby("ts_code", group_keys=False)

    out["ret_close_1d"] = by_symbol["close"].pct_change(1)
    out["ret_close_3d"] = by_symbol["close"].pct_change(3)
    out["ret_close_5d"] = by_symbol["close"].pct_change(5)
    out["close_ma5_ratio"] = close_series / by_symbol["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(5, min_periods=5).mean()) - 1.0
    out["close_ma10_ratio"] = close_series / by_symbol["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(10, min_periods=10).mean()) - 1.0
    out["close_vol_5d"] = by_symbol["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").pct_change().rolling(5, min_periods=5).std())
    out["overnight_prev_1d"] = by_symbol["overnight_return_open"].shift(1)
    out["overnight_prev_3d_mean"] = by_symbol["overnight_return_open"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(3, min_periods=3).mean())
    out["overnight_prev_5d_mean"] = by_symbol["overnight_return_open"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(5, min_periods=5).mean())
    out["overnight_prev_5d_std"] = by_symbol["overnight_return_open"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).rolling(5, min_periods=5).std())
    out["overnight_positive_rate_5d"] = by_symbol["overnight_return_open"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).gt(0).rolling(5, min_periods=5).mean())
    out["gap_days"] = pd.to_numeric(out.get("gap_days"), errors="coerce")
    out["next_open_gap"] = next_open_series / close_series.replace(0, pd.NA) - 1.0

    roll_min_5 = by_symbol["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(5, min_periods=5).min())
    roll_max_5 = by_symbol["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(5, min_periods=5).max())
    roll_max_10 = by_symbol["close"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(10, min_periods=10).max())
    range_5 = (roll_max_5 - roll_min_5).replace(0, pd.NA)
    out["close_range_pos_5d"] = (close_series - roll_min_5) / range_5
    out["close_drawdown_10d"] = close_series / roll_max_10 - 1.0

    prev_limit = by_symbol["is_limit_move_like"].shift(1) if "is_limit_move_like" in out.columns else pd.Series(pd.NA, index=out.index)
    prev_soft = by_symbol["is_soft_outlier"].shift(1) if "is_soft_outlier" in out.columns else pd.Series(pd.NA, index=out.index)
    out["prev_limit_move_like_1d"] = prev_limit.astype("boolean")
    out["prev_soft_outlier_1d"] = prev_soft.astype("boolean")

    list_date = pd.to_datetime(out.get("list_date"), errors="coerce")
    trade_date = pd.to_datetime(out.get("trade_date"), errors="coerce")
    out["days_since_listed"] = (trade_date - list_date).dt.days
    out["is_new_listing_180d"] = out["days_since_listed"] < 180

    total_mv_series = pd.to_numeric(out["total_mv"], errors="coerce") if "total_mv" in out.columns else pd.Series(float("nan"), index=out.index, dtype="float64")
    circ_mv_series = pd.to_numeric(out["circ_mv"], errors="coerce") if "circ_mv" in out.columns else pd.Series(float("nan"), index=out.index, dtype="float64")
    out["log_total_mv"] = total_mv_series.map(lambda x: math.log1p(x) if pd.notna(x) and x >= 0 else pd.NA)
    out["log_circ_mv"] = circ_mv_series.map(lambda x: math.log1p(x) if pd.notna(x) and x >= 0 else pd.NA)
    out["net_mf_amount"] = pd.to_numeric(out["net_mf_amount"], errors="coerce") if "net_mf_amount" in out.columns else pd.Series(float("nan"), index=out.index, dtype="float64")
    amount_t = pd.to_numeric(out["amount_t"], errors="coerce") if "amount_t" in out.columns else pd.Series(float("nan"), index=out.index, dtype="float64")
    out["net_mf_ratio"] = out["net_mf_amount"] / amount_t.replace(0, pd.NA)

    if "minute_vol_30m" in out.columns and "vol_t" in out.columns:
        out["minute_vol_share_30m"] = pd.to_numeric(out.get("minute_vol_30m"), errors="coerce") / pd.to_numeric(out.get("vol_t"), errors="coerce").replace(0, pd.NA)
    if "minute_amount_30m" in out.columns and "amount_t" in out.columns:
        out["minute_amount_share_30m"] = pd.to_numeric(out.get("minute_amount_30m"), errors="coerce") / pd.to_numeric(out.get("amount_t"), errors="coerce").replace(0, pd.NA)

    return out


SCORE_SPECS = [
    ("ret_close_1d", 0.16, True),
    ("ret_close_3d", 0.10, True),
    ("ret_close_5d", 0.06, True),
    ("close_ma5_ratio", 0.08, False),
    ("close_ma10_ratio", 0.06, False),
    ("close_range_pos_5d", 0.12, True),
    ("close_drawdown_10d", 0.10, True),
    ("close_vol_5d", 0.08, False),
    ("overnight_prev_1d", 0.06, False),
    ("overnight_prev_3d_mean", 0.06, False),
    ("overnight_prev_5d_mean", 0.05, False),
    ("overnight_prev_5d_std", 0.03, False),
    ("overnight_positive_rate_5d", 0.05, False),
    ("minute_last30_return", 0.04, True),
    ("minute_range_pos_30m", 0.03, True),
    ("minute_vwap_gap_30m", 0.03, True),
    ("net_mf_ratio", 0.03, True),
    ("volume_ratio", 0.02, True),
    ("gap_days", 0.03, False),
    ("is_new_listing_180d", 0.02, False),
    ("prev_limit_move_like_1d", 0.02, True),
    ("prev_soft_outlier_1d", 0.02, True),
]


def add_baseline_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["overnight_score"] = 0.0

    for col, weight, ascending in SCORE_SPECS:
        if col not in out.columns:
            continue
        series = out[col]
        if series.dtype == bool:
            series = series.astype(float)
        values = pd.to_numeric(series, errors="coerce")
        ranks = values.groupby(out["trade_date"]).rank(pct=True, ascending=ascending)
        out[f"score_component__{col}"] = ranks
        out["overnight_score"] = out["overnight_score"] + weight * ranks.fillna(0.5)

    penalty = pd.Series(0.0, index=out.index)
    for flag, pen in [("is_long_gap", 0.20), ("is_limit_move_like", 0.20), ("is_soft_outlier", 0.08), ("is_extreme", 0.20)]:
        if flag in out.columns:
            raw = out[flag].astype(str).str.lower().map({"true": 1.0, "false": 0.0}).fillna(0.0)
            penalty += raw * pen
    out["overnight_score"] = out["overnight_score"] - penalty
    out["rank_in_day"] = out.groupby("trade_date")["overnight_score"].rank(method="first", ascending=False)
    return out


def build_topn_input(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    keep_cols = [c for c in [
        "trade_date", "next_trade_date", "ts_code", "name", "industry", "market",
        "overnight_score", "rank_in_day", "overnight_return_open",
        "overnight_return_0935", "overnight_return_0945", "overnight_return_1000",
        "close", "next_open", "next_close_0935", "next_close_0945", "next_close_1000", "gap_days",
        "ret_close_1d", "ret_close_3d", "ret_close_5d",
        "close_ma5_ratio", "close_ma10_ratio", "close_vol_5d",
        "close_range_pos_5d", "close_drawdown_10d",
        "overnight_prev_1d", "overnight_prev_3d_mean", "overnight_prev_5d_mean", "overnight_prev_5d_std",
        "overnight_positive_rate_5d",
        "turnover_rate", "volume_ratio", "pe", "pb", "total_mv", "circ_mv",
        "net_mf_amount", "net_mf_ratio",
        "minute_last30_return", "minute_last15_return", "minute_range_pos_30m", "minute_vwap_gap_30m",
        "minute_vol_share_30m", "minute_amount_share_30m",
        "is_new_listing_180d", "prev_limit_move_like_1d", "prev_soft_outlier_1d",
        "is_long_gap", "is_limit_move_like", "is_soft_outlier", "is_extreme",
    ] if c in df.columns]
    ranked = df.sort_values(["trade_date", "rank_in_day", "ts_code"]).copy()
    ranked["selected_top_n"] = ranked["rank_in_day"] <= top_n
    ranked = ranked.loc[ranked["selected_top_n"]].copy()
    return ranked[keep_cols + ["selected_top_n"]].reset_index(drop=True)


def collect_symbol_frames(
    symbols: Iterable[str],
    start_date: str,
    end_date: str,
    cache_root: Path,
    force_refresh: bool = False,
    labels: pd.DataFrame | None = None,
    include_minute_features: bool = False,
    minute_start_time: str = "14:30:00",
    minute_end_time: str = "15:00:00",
    minute_freq: str = "5min",
    minute_max_symbol_dates: int = 0,
    minute_cache_dir: Path | None = None,
    minute_cache_only: bool = False,
    fetch_tushare_factors: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    errors: list[str] = []
    frames: dict[str, list[pd.DataFrame]] = {"daily_basic": [], "moneyflow": [], "stk_factor": []}

    for api_name in ["daily_basic", "moneyflow", "stk_factor"]:
        api_cache = cache_root / api_name
        for ts_code in symbols:
            df, err = fetch_symbol_frame(
                api_name=api_name,
                ts_code=str(ts_code),
                start_date=start_date,
                end_date=end_date,
                cache_dir=api_cache,
                force_refresh=force_refresh,
                allow_remote=fetch_tushare_factors,
            )
            if err and err not in {"empty", "cache_miss_remote_disabled"}:
                errors.append(f"{api_name} {ts_code}: {err}")
            if df is not None and not df.empty:
                frames[api_name].append(df)

    minute_feature_df = pd.DataFrame()
    if include_minute_features and labels is not None and not labels.empty:
        minute_rows: list[dict[str, object]] = []
        minute_cache = Path(minute_cache_dir) if minute_cache_dir is not None else (cache_root / DEFAULT_MINUTE_CACHE_DIR)
        pairs = labels[["ts_code", "trade_date"]].dropna().drop_duplicates().sort_values(["trade_date", "ts_code"])
        if minute_max_symbol_dates and minute_max_symbol_dates > 0:
            pairs = pairs.head(int(minute_max_symbol_dates))
        for row in pairs.itertuples(index=False):
            mins, err = fetch_minute_window_frame(
                ts_code=str(row.ts_code),
                trade_date=str(row.trade_date),
                cache_dir=minute_cache,
                start_time=minute_start_time,
                end_time=minute_end_time,
                freq=minute_freq,
                force_refresh=force_refresh,
                allow_remote=not minute_cache_only,
            )
            if err and err not in {"empty", "cache_miss"}:
                errors.append(f"minute {row.ts_code} {row.trade_date}: {err}")
            minute_rows.append(summarize_minute_features(mins, str(row.ts_code), str(row.trade_date)))
        if minute_rows:
            minute_feature_df = pd.DataFrame(minute_rows)

    def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
        if not parts:
            return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        if "trade_date" in out.columns:
            out["trade_date"] = out["trade_date"].map(_normalize_date)
        return out

    return _concat(frames["daily_basic"]), _concat(frames["moneyflow"]), _concat(frames["stk_factor"]), minute_feature_df, errors


def write_audit(
    path: Path,
    labels: pd.DataFrame,
    features: pd.DataFrame,
    topn: pd.DataFrame,
    errors: list[str],
    top_n: int,
    source_labels: Path,
    fetch_tushare_factors: bool,
) -> None:
    _ensure_dir(path.parent)
    missing_rates = []
    for col in [
        "ret_close_1d", "ret_close_3d", "ret_close_5d",
        "close_ma5_ratio", "close_ma10_ratio", "close_vol_5d",
        "close_range_pos_5d", "close_drawdown_10d",
        "overnight_prev_1d", "overnight_prev_3d_mean", "overnight_prev_5d_mean", "overnight_prev_5d_std",
        "overnight_positive_rate_5d",
        "minute_last30_return", "minute_last15_return", "minute_range_pos_30m", "minute_vwap_gap_30m",
        "minute_vol_share_30m", "minute_amount_share_30m",
        "volume_ratio", "turnover_rate", "net_mf_ratio", "log_total_mv", "log_circ_mv",
        "is_new_listing_180d", "prev_limit_move_like_1d", "prev_soft_outlier_1d", "industry", "market"
    ]:
        if col in features.columns:
            ratio = float(features[col].isna().mean())
            missing_rates.append(f"- `{col}` missing rate: `{ratio:.4f}`")

    text = f"""# Overnight Feature Build Audit

- Source labels: `{source_labels}`
- Label rows used: `{len(labels)}`
- Feature rows written: `{len(features)}`
- Top-{top_n} rows written: `{len(topn)}`
- Trade dates covered: `{labels['trade_date'].min()}` -> `{labels['trade_date'].max()}`
- Symbols covered: `{labels['ts_code'].nunique()}`
- Selected days in Top-{top_n}: `{topn['trade_date'].nunique() if not topn.empty else 0}`
- Tushare factor fetch mode: `{'remote enabled' if fetch_tushare_factors else 'safe cache-only; daily_basic/moneyflow/stk_factor remote calls disabled'}`

## Missingness snapshot
{chr(10).join(missing_rates) if missing_rates else '- no tracked feature columns found'}

## Upstream fetch warnings
"""
    if errors:
        text += "\n".join(f"- `{msg}`" for msg in errors[:200])
        if len(errors) > 200:
            text += f"\n- ... truncated {len(errors) - 200} additional warnings"
    else:
        text += "- none"

    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified overnight feature table and Top-N baseline input")
    parser.add_argument("--labels", default=str(DEFAULT_LABELS), help="Path to clean overnight label CSV")
    parser.add_argument("--start-date", default=DEFAULT_LOOKBACK_START, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_LOOKBACK_END, help="YYYY-MM-DD")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top-N picks per day")
    parser.add_argument("--limit-symbols", type=int, default=0, help="Optional cap for fast smoke tests")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore cached API CSVs. Note: high-friction Tushare factors still require --fetch-tushare-factors to call remote APIs")
    parser.add_argument("--fetch-tushare-factors", action="store_true", help="Explicitly fetch/refresh daily_basic, moneyflow and stk_factor from Tushare; default only uses local cached factor CSVs and never calls these APIs")
    parser.add_argument("--include-minute-features", action="store_true", help="Fetch/cache same-day 14:30-15:00 minute bars and derive tail-session features")
    parser.add_argument("--minute-start-time", default="14:30:00", help="Minute feature window start time HH:MM[:SS]")
    parser.add_argument("--minute-end-time", default="15:00:00", help="Minute feature window end time HH:MM[:SS]")
    parser.add_argument("--minute-max-symbol-dates", type=int, default=0, help="Optional cap on symbol-date minute fetches for smoke tests")
    parser.add_argument("--minute-cache-dir", default=str(DEFAULT_MINUTE_CACHE_DIR), help="Minute cache directory, relative to repo unless absolute")
    parser.add_argument("--minute-cache-only", action="store_true", help="Use only prefetched minute cache and do not hit remote minute API during feature build")
    args = parser.parse_args()

    labels_path = Path(args.labels)
    start_date = _normalize_date(args.start_date)
    end_date = _normalize_date(args.end_date)

    labels = load_labels(labels_path, start_date, end_date)
    symbols = sorted(labels["ts_code"].dropna().astype(str).unique().tolist())
    if args.limit_symbols and args.limit_symbols > 0:
        symbols = symbols[: args.limit_symbols]
        labels = labels.loc[labels["ts_code"].isin(symbols)].copy()

    out_root = DEFAULT_OUT_ROOT
    feature_dir = out_root / "features"
    backtest_dir = out_root / "backtest_inputs"
    audit_dir = out_root / "audit"
    cache_root = out_root / "cache"
    for p in [feature_dir, backtest_dir, audit_dir, cache_root]:
        _ensure_dir(p)

    stock_basic = fetch_stock_basic(cache_root / "stock_basic", force_refresh=args.force_refresh)
    minute_cache_dir = Path(args.minute_cache_dir)
    if not minute_cache_dir.is_absolute():
        minute_cache_dir = Path.cwd() / minute_cache_dir
    daily_basic, moneyflow, stk_factor, minute_features, errors = collect_symbol_frames(
        symbols,
        start_date,
        end_date,
        cache_root,
        force_refresh=args.force_refresh,
        labels=labels,
        include_minute_features=args.include_minute_features,
        minute_start_time=args.minute_start_time,
        minute_end_time=args.minute_end_time,
        minute_max_symbol_dates=args.minute_max_symbol_dates,
        minute_cache_dir=minute_cache_dir,
        minute_cache_only=args.minute_cache_only,
        fetch_tushare_factors=args.fetch_tushare_factors,
    )

    merged = merge_feature_frames(labels, stock_basic, daily_basic, moneyflow, stk_factor, minute_features)
    features = add_derived_features(merged)
    scored = add_baseline_scores(features)
    topn = build_topn_input(scored, args.top_n)

    suffix = f"{_fmt_date(start_date)}_{_fmt_date(end_date)}"
    feature_path = feature_dir / f"overnight_features_{suffix}.csv"
    topn_path = backtest_dir / f"topn_baseline_input_{suffix}.csv"
    audit_path = audit_dir / f"overnight_feature_build_{suffix}.md"

    scored.to_csv(feature_path, index=False)
    topn.to_csv(topn_path, index=False)
    write_audit(audit_path, labels, scored, topn, errors, args.top_n, labels_path, args.fetch_tushare_factors)

    print(f"Wrote feature table: {feature_path} rows={len(scored)}")
    print(f"Wrote Top-{args.top_n} input: {topn_path} rows={len(topn)}")
    print(f"Wrote audit: {audit_path}")
    if errors:
        print(f"Warnings: {len(errors)} upstream fetch issues (see audit)")


if __name__ == "__main__":
    main()
