from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from dataflows.tushare_provider import TushareRateLimitError, _fmt_date, _get_pro, _parse_date, _safe_call, _to_ts_code


DEFAULT_MINUTE_CACHE_DIR = Path("data/overnight_mvp/cache/minute_1430_features")


def normalize_date(value: str) -> str:
    return _parse_date(str(value).strip())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def is_rate_limit_error(message: str) -> bool:
    msg = str(message)
    return any(kw in msg for kw in ["频率超限", "每分钟", "每小时", "rate limit", "2次/秒", "2次/分钟", "2次/天", "1次/小时"])


def _bare_code(ts_code: str) -> str:
    return _to_ts_code(ts_code).split(".", 1)[0]


def _eastmoney_secid(ts_code: str) -> str:
    code, exch = _to_ts_code(ts_code).split(".", 1)
    market = "1" if exch == "SH" else "0"
    return f"{market}.{code}"


def _fetch_eastmoney_minute_window(
    ts_code: str,
    trade_date: str,
    start_time: str,
    end_time: str,
    freq: str,
    timeout: float = 8.0,
) -> pd.DataFrame:
    """Fetch A-share intraday minute bars from Eastmoney trends endpoint.

    This is intentionally used before Tushare stk_mins in the live 14:30-14:57
    window because stk_mins may be limited to 1 call/hour on some accounts.
    Eastmoney returns 1-minute bars for recent days; for coarser freq we resample
    locally to the requested window.
    """
    import requests

    secid = _eastmoney_secid(ts_code)
    url = "https://push2his.eastmoney.com/api/qt/stock/trends2/get"
    params = {
        "fields1": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58",
        "ndays": "2",
        "iscr": "0",
        "iscca": "0",
        "secid": secid,
    }
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://quote.eastmoney.com/"}
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            break
        except Exception:
            if attempt >= 2:
                raise
            time.sleep(0.8 * (attempt + 1))
    trends = (payload.get("data") or {}).get("trends") or []
    rows: list[dict[str, object]] = []
    day = normalize_date(trade_date)
    start_label = f"{day} {start_time}"
    end_label = f"{day} {end_time}"
    for line in trends:
        parts = str(line).split(",")
        if len(parts) < 7:
            continue
        trade_time = parts[0]
        if trade_time < start_label or trade_time > end_label:
            continue
        rows.append({
            "ts_code": _to_ts_code(ts_code),
            "trade_time": pd.to_datetime(trade_time, errors="coerce"),
            "open": pd.to_numeric(parts[1], errors="coerce"),
            "close": pd.to_numeric(parts[2], errors="coerce"),
            "high": pd.to_numeric(parts[3], errors="coerce"),
            "low": pd.to_numeric(parts[4], errors="coerce"),
            "vol": pd.to_numeric(parts[5], errors="coerce"),
            "amount": pd.to_numeric(parts[6], errors="coerce"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.dropna(subset=["trade_time"]).sort_values("trade_time").reset_index(drop=True)
    if str(freq).lower() in {"1min", "1m", "1"}:
        return df

    # Tushare stk_mins freq labels are usually 1min/5min/15min/etc.  Resample
    # from Eastmoney 1-minute bars when a coarser cache is requested.
    freq_map = {"5min": "5min", "5m": "5min", "15min": "15min", "15m": "15min", "30min": "30min", "30m": "30min"}
    rule = freq_map.get(str(freq).lower())
    if not rule:
        return df
    d = df.set_index("trade_time")
    resampled = d.resample(rule, label="right", closed="right").agg({
        "ts_code": "last",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "vol": "sum",
        "amount": "sum",
    }).dropna(subset=["open", "close"]).reset_index()
    return resampled[["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]]


def minute_cache_path(
    cache_dir: Path,
    ts_code: str,
    trade_date: str,
    freq: str,
    start_time: str,
    end_time: str,
) -> Path:
    safe = str(ts_code).replace("/", "_")
    day = _fmt_date(normalize_date(trade_date))
    start_label = str(start_time).replace(":", "")[:4]
    end_label = str(end_time).replace(":", "")[:4]
    freq_label = str(freq).replace("/", "_")
    return Path(cache_dir) / f"minute_{safe}_{day}_{start_label}_{end_label}_{freq_label}.csv"


def minute_meta_path(
    cache_dir: Path,
    ts_code: str,
    trade_date: str,
    freq: str,
    start_time: str,
    end_time: str,
) -> Path:
    return minute_cache_path(cache_dir, ts_code, trade_date, freq, start_time, end_time).with_suffix(".meta.json")


def load_cached_minute_window_frame(
    ts_code: str,
    trade_date: str,
    cache_dir: Path,
    start_time: str = "14:30:00",
    end_time: str = "15:00:00",
    freq: str = "5min",
) -> tuple[pd.DataFrame, str | None]:
    cache_path = minute_cache_path(cache_dir, ts_code, trade_date, freq, start_time, end_time)
    if not cache_path.exists():
        return pd.DataFrame(), "cache_miss"
    try:
        df = pd.read_csv(cache_path)
        if "trade_time" in df.columns:
            df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
        return df, None
    except Exception as exc:
        return pd.DataFrame(), f"cache_read_error: {type(exc).__name__}: {exc}"


def _write_meta(path: Path, payload: dict[str, object]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def fetch_minute_window_frame(
    ts_code: str,
    trade_date: str,
    cache_dir: Path,
    start_time: str = "14:30:00",
    end_time: str = "15:00:00",
    freq: str = "5min",
    force_refresh: bool = False,
    max_retries: int = 2,
    allow_remote: bool = True,
    write_meta: bool = True,
    source: str = "eastmoney,tushare",
) -> tuple[pd.DataFrame, str | None]:
    cache_dir = Path(cache_dir)
    ensure_dir(cache_dir)
    trade_date = normalize_date(trade_date)
    cache_path = minute_cache_path(cache_dir, ts_code, trade_date, freq, start_time, end_time)
    meta_path = minute_meta_path(cache_dir, ts_code, trade_date, freq, start_time, end_time)

    if cache_path.exists() and not force_refresh:
        df, err = load_cached_minute_window_frame(ts_code, trade_date, cache_dir, start_time=start_time, end_time=end_time, freq=freq)
        if err is None:
            return df, None

    if not allow_remote:
        return pd.DataFrame(), "cache_miss"

    started_at = time.time()
    sources = [s.strip().lower() for s in str(source or "tushare").split(",") if s.strip()]
    if not sources:
        sources = ["tushare"]

    last_err: str | None = None

    if "eastmoney" in sources:
        try:
            df = _fetch_eastmoney_minute_window(ts_code, trade_date, start_time, end_time, freq)
            if df is not None and not df.empty:
                df = df.copy()
                if "trade_time" in df.columns:
                    df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
                keep = [c for c in ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"] if c in df.columns]
                df = df[keep].sort_values("trade_time").reset_index(drop=True)
                df.to_csv(cache_path, index=False)
                if write_meta:
                    _write_meta(meta_path, {
                        "ts_code": ts_code,
                        "trade_date": trade_date,
                        "freq": freq,
                        "start_time": start_time,
                        "end_time": end_time,
                        "status": "ok",
                        "source": "eastmoney",
                        "row_count": int(len(df)),
                        "elapsed_seconds": round(time.time() - started_at, 3),
                        "cache_path": str(cache_path),
                    })
                return df, None
            last_err = "eastmoney_empty"
        except Exception as exc:
            last_err = f"eastmoney:{type(exc).__name__}: {exc}"

    if "tushare" not in sources:
        if write_meta:
            _write_meta(meta_path, {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "freq": freq,
                "start_time": start_time,
                "end_time": end_time,
                "status": "empty" if last_err == "eastmoney_empty" else "error",
                "source": ",".join(sources),
                "error": last_err or "no_enabled_source",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "cache_path": str(cache_path),
            })
        return pd.DataFrame(), last_err or "empty"

    try:
        pro = _get_pro()
    except Exception as exc:
        last_err = f"tushare_init:{type(exc).__name__}: {exc}"
        if write_meta:
            _write_meta(meta_path, {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "freq": freq,
                "start_time": start_time,
                "end_time": end_time,
                "status": "error",
                "source": "tushare",
                "error": last_err,
                "elapsed_seconds": round(time.time() - started_at, 3),
                "cache_path": str(cache_path),
            })
        return pd.DataFrame(), last_err
    start_dt = f"{trade_date} {start_time}"
    end_dt = f"{trade_date} {end_time}"

    for attempt in range(max_retries + 1):
        try:
            df = _safe_call(
                pro.stk_mins,
                ts_code=_to_ts_code(ts_code),
                start_date=start_dt,
                end_date=end_dt,
                freq=freq,
            )
            break
        except TushareRateLimitError as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if write_meta:
                _write_meta(meta_path, {
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "freq": freq,
                    "start_time": start_time,
                    "end_time": end_time,
                    "status": "rate_limited",
                    "source": "tushare",
                    "error": last_err,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                    "cache_path": str(cache_path),
                    "fast_fuse": True,
                })
            return pd.DataFrame(), last_err
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if is_rate_limit_error(exc):
                if write_meta:
                    _write_meta(meta_path, {
                        "ts_code": ts_code,
                        "trade_date": trade_date,
                        "freq": freq,
                        "start_time": start_time,
                        "end_time": end_time,
                        "status": "rate_limited",
                        "source": "tushare",
                        "error": last_err,
                        "elapsed_seconds": round(time.time() - started_at, 3),
                        "cache_path": str(cache_path),
                        "fast_fuse": True,
                    })
                return pd.DataFrame(), last_err
            if write_meta:
                _write_meta(meta_path, {
                    "ts_code": ts_code,
                    "trade_date": trade_date,
                    "freq": freq,
                    "start_time": start_time,
                    "end_time": end_time,
                    "status": "error",
                    "source": "tushare",
                    "error": last_err,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                    "cache_path": str(cache_path),
                })
            return pd.DataFrame(), last_err

    if df is None or df.empty:
        if write_meta:
            _write_meta(meta_path, {
                "ts_code": ts_code,
                "trade_date": trade_date,
                "freq": freq,
                "start_time": start_time,
                "end_time": end_time,
                "status": "empty",
                "source": "tushare",
                "elapsed_seconds": round(time.time() - started_at, 3),
                "cache_path": str(cache_path),
            })
        return pd.DataFrame(), "empty"

    df = df.copy()
    if "trade_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["trade_time"], errors="coerce")
    keep = [c for c in ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"] if c in df.columns]
    df = df[keep].sort_values("trade_time").reset_index(drop=True)
    df.to_csv(cache_path, index=False)
    if write_meta:
        _write_meta(meta_path, {
            "ts_code": ts_code,
            "trade_date": trade_date,
            "freq": freq,
            "start_time": start_time,
            "end_time": end_time,
            "status": "ok",
            "source": "tushare",
            "row_count": int(len(df)),
            "elapsed_seconds": round(time.time() - started_at, 3),
            "cache_path": str(cache_path),
        })
    return df, None


def summarize_minute_features(mins: pd.DataFrame, ts_code: str, trade_date: str) -> dict[str, object]:
    record: dict[str, object] = {"ts_code": ts_code, "trade_date": normalize_date(trade_date)}
    if mins is None or mins.empty:
        record.update({
            "minute_bar_count_30m": 0,
            "minute_last30_return": pd.NA,
            "minute_last15_return": pd.NA,
            "minute_range_pos_30m": pd.NA,
            "minute_vwap_gap_30m": pd.NA,
            "minute_vol_30m": pd.NA,
            "minute_amount_30m": pd.NA,
        })
        return record

    data = mins.copy().sort_values("trade_time").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    first_open = pd.to_numeric(data.iloc[0].get("open"), errors="coerce")
    last_close = pd.to_numeric(data.iloc[-1].get("close"), errors="coerce")
    low_30m = pd.to_numeric(data.get("low"), errors="coerce").min()
    high_30m = pd.to_numeric(data.get("high"), errors="coerce").max()
    vol_30m = pd.to_numeric(data.get("vol"), errors="coerce").sum(min_count=1)
    amount_30m = pd.to_numeric(data.get("amount"), errors="coerce").sum(min_count=1)

    last15 = data.loc[data["trade_time"].dt.strftime("%H:%M:%S") >= "14:45:00"].copy() if "trade_time" in data.columns else pd.DataFrame()
    last15_open = pd.to_numeric(last15.iloc[0].get("open"), errors="coerce") if not last15.empty else pd.NA

    vwap_30m = pd.NA
    if pd.notna(vol_30m) and float(vol_30m) > 0 and pd.notna(amount_30m):
        vwap_30m = float(amount_30m) / float(vol_30m)

    range_pos = pd.NA
    if pd.notna(high_30m) and pd.notna(low_30m) and float(high_30m) != float(low_30m):
        range_pos = (float(last_close) - float(low_30m)) / (float(high_30m) - float(low_30m))

    record.update({
        "minute_bar_count_30m": int(len(data)),
        "minute_last30_return": None if pd.isna(first_open) or float(first_open) == 0 or pd.isna(last_close) else float(last_close) / float(first_open) - 1.0,
        "minute_last15_return": None if pd.isna(last15_open) or float(last15_open) == 0 or pd.isna(last_close) else float(last_close) / float(last15_open) - 1.0,
        "minute_range_pos_30m": range_pos,
        "minute_vwap_gap_30m": None if pd.isna(vwap_30m) or float(vwap_30m) == 0 or pd.isna(last_close) else float(last_close) / float(vwap_30m) - 1.0,
        "minute_vol_30m": None if pd.isna(vol_30m) else float(vol_30m),
        "minute_amount_30m": None if pd.isna(amount_30m) else float(amount_30m),
    })
    return record
