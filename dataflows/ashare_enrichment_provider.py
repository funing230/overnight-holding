from __future__ import annotations

"""A-share V3.1 style enrichment adapters for overnight live workflow.

The functions here are intentionally defensive: public data endpoints often
change columns or occasionally fail.  Every fetcher returns a normalized feature
frame with ts_code and a compact set of numeric/string signals.  Missing vendors
produce neutral rows instead of breaking the live decision pipeline.
"""

import json
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from dataflows.realtime_snapshot_provider import normalize_ts_code, ts_code_to_realtime_symbol


SH_MARKETS = ("5", "6", "9")


def _bare(ts_code: str) -> str:
    return ts_code_to_realtime_symbol(normalize_ts_code(ts_code))


def _tx_symbol(ts_code: str) -> str:
    code = _bare(ts_code)
    return ("sh" if code.startswith(SH_MARKETS) else "sz") + code


def _ak_market(ts_code: str) -> str:
    code = _bare(ts_code)
    return "sh" if code.startswith(SH_MARKETS) else "sz"


def _sina_symbol(ts_code: str) -> str:
    code = _bare(ts_code)
    return ("sh" if code.startswith(SH_MARKETS) else "sz") + code


def _num(v, default=pd.NA):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def _ensure_ts_rows(ts_codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"ts_code": [normalize_ts_code(x) for x in ts_codes]})


def _safe_to_csv(df: pd.DataFrame, path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return str(p)


# ---------------------------------------------------------------------------
# 1) Tencent realtime snapshot: primary HTTP source for 14:55 snapshot.
# ---------------------------------------------------------------------------


def fetch_tencent_realtime_snapshot(ts_codes: list[str], chunk_size: int = 80, timeout: float = 8.0) -> pd.DataFrame:
    """Fetch realtime quotes from Tencent qt.gtimg.cn and normalize columns.

    Output schema is compatible with overnight_live_provider.load_snapshot_csv.
    Tushare is no longer required for the primary snapshot path; callers can use
    fetch_realtime_snapshot_with_fallback(..., primary="tencent", fallback="tushare").
    """
    normalized = [normalize_ts_code(x) for x in ts_codes]
    rows: list[dict[str, object]] = []
    run_ts = datetime.now().isoformat(timespec="seconds")
    for i in range(0, len(normalized), max(1, int(chunk_size))):
        chunk = normalized[i : i + max(1, int(chunk_size))]
        q = ",".join(_tx_symbol(x) for x in chunk)
        url = f"https://qt.gtimg.cn/q={q}"
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("gbk", errors="ignore")
        for line in text.split(";\n"):
            line = line.strip().strip(";")
            if not line or "=\"" not in line:
                continue
            prefix, payload = line.split("=\"", 1)
            payload = payload.rstrip('"')
            fields = payload.split("~")
            if len(fields) < 6:
                continue
            tx = prefix.split("v_", 1)[-1]
            code = tx[-6:]
            ts_code = normalize_ts_code(code)
            row = {
                "ts_code": ts_code,
                "code": code,
                "name": fields[1] if len(fields) > 1 else "",
                "last_price": _num(fields[3] if len(fields) > 3 else None),
                "pre_close": _num(fields[4] if len(fields) > 4 else None),
                "open": _num(fields[5] if len(fields) > 5 else None),
                "volume": _num(fields[6] if len(fields) > 6 else None),
                "amount": _num(fields[37] if len(fields) > 37 else None),
                "quote_time": fields[30] if len(fields) > 30 else "",
                "pct_change": _num(fields[32] if len(fields) > 32 else None),
                "high": _num(fields[33] if len(fields) > 33 else None),
                "low": _num(fields[34] if len(fields) > 34 else None),
                "turnover_rate": _num(fields[38] if len(fields) > 38 else None),
                "pe_ttm": _num(fields[39] if len(fields) > 39 else None),
                "pb": _num(fields[46] if len(fields) > 46 else None),
                "source": "tencent.qt.gtimg.cn",
                "run_ts": run_ts,
            }
            # Tencent quote_time is often yyyymmddHHMMSS. Keep raw plus date.
            qt = str(row.get("quote_time") or "")
            if len(qt) >= 8 and qt[:8].isdigit():
                row["quote_date"] = f"{qt[:4]}-{qt[4:6]}-{qt[6:8]}"
                if len(qt) >= 14:
                    row["quote_time"] = f"{qt[8:10]}:{qt[10:12]}:{qt[12:14]}"
            rows.append(row)
        time.sleep(0.05)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.loc[out["ts_code"].isin(normalized)].drop_duplicates("ts_code", keep="last").reset_index(drop=True)
    for col in ["last_price", "open", "high", "low", "pre_close", "volume", "amount", "pct_change", "turnover_rate", "pe_ttm", "pb"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def fetch_realtime_snapshot_with_fallback(ts_codes: list[str], primary: str = "tencent", fallback: str = "tushare", chunk_size: int = 80) -> pd.DataFrame:
    from dataflows.realtime_snapshot_provider import fetch_tushare_realtime_snapshot

    errors: list[str] = []
    normalized = [normalize_ts_code(x) for x in ts_codes]
    primary_df = pd.DataFrame()
    if primary == "tencent":
        try:
            primary_df = fetch_tencent_realtime_snapshot(normalized, chunk_size=chunk_size)
        except Exception as exc:
            errors.append(f"tencent:{type(exc).__name__}:{exc}")
    elif primary == "tushare":
        try:
            primary_df = fetch_tushare_realtime_snapshot(normalized, chunk_size=chunk_size)
        except Exception as exc:
            errors.append(f"tushare:{type(exc).__name__}:{exc}")
    else:
        raise ValueError(f"unsupported primary source: {primary}")

    got = set(primary_df.get("ts_code", pd.Series(dtype=str)).astype(str)) if not primary_df.empty else set()
    missing = [x for x in normalized if x not in got]
    frames = [primary_df] if not primary_df.empty else []
    if missing and fallback:
        try:
            if fallback == "tushare":
                fb = fetch_tushare_realtime_snapshot(missing, chunk_size=300)
            elif fallback == "tencent":
                fb = fetch_tencent_realtime_snapshot(missing, chunk_size=chunk_size)
            else:
                raise ValueError(f"unsupported fallback source: {fallback}")
            if not fb.empty:
                fb = fb.copy()
                fb["source"] = fb.get("source", fallback).astype(str) + "+fallback"
                frames.append(fb)
        except Exception as exc:
            errors.append(f"{fallback}:{type(exc).__name__}:{exc}")
    if not frames:
        out = pd.DataFrame(columns=["ts_code", "last_price", "source"])
    else:
        out = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code", keep="first").reset_index(drop=True)
    if errors:
        out.attrs["errors"] = errors
    return out


# ---------------------------------------------------------------------------
# 3-8) Candidate enrichment features.
# ---------------------------------------------------------------------------


def fetch_fund_flow_features(ts_codes: list[str]) -> pd.DataFrame:
    rows = []
    try:
        import akshare as ak
    except Exception as exc:
        out = _ensure_ts_rows(ts_codes)
        out["ashare_enrich_error"] = f"akshare_missing:{exc}"
        return out
    for ts_code in ts_codes:
        code = _bare(ts_code)
        rec = {"ts_code": normalize_ts_code(ts_code)}
        try:
            df = ak.stock_individual_fund_flow(stock=code, market=_ak_market(ts_code))
            if df is not None and not df.empty:
                tail = df.tail(5).copy()
                latest = tail.iloc[-1]
                for src, dst in [
                    ("主力净流入-净额", "fund_main_net_inflow_1d"),
                    ("主力净流入-净占比", "fund_main_net_inflow_ratio_1d"),
                    ("超大单净流入-净额", "fund_super_net_inflow_1d"),
                    ("大单净流入-净额", "fund_large_net_inflow_1d"),
                    ("小单净流入-净额", "fund_small_net_inflow_1d"),
                    ("小单净流入-净占比", "fund_small_net_inflow_ratio_1d"),
                ]:
                    rec[dst] = _num(latest.get(src)) if src in latest.index else pd.NA
                if "主力净流入-净额" in tail.columns:
                    s = pd.to_numeric(tail["主力净流入-净额"], errors="coerce")
                    rec["fund_main_net_inflow_3d_sum"] = float(s.tail(3).sum())
                    rec["fund_main_net_inflow_5d_sum"] = float(s.tail(5).sum())
                    rec["fund_main_positive_days_5d"] = int((s.tail(5) > 0).sum())
        except Exception as exc:
            rec["fund_flow_error"] = f"{type(exc).__name__}:{exc}"
        rows.append(rec)
        time.sleep(0.08)
    return pd.DataFrame(rows)


def fetch_akshare_minute_features(ts_codes: list[str], trade_date: str, start_time: str = "14:30:00", end_time: str = "15:00:00") -> pd.DataFrame:
    rows = []
    try:
        import akshare as ak
    except Exception as exc:
        out = _ensure_ts_rows(ts_codes)
        out["minute_error"] = f"akshare_missing:{exc}"
        return out
    day = str(trade_date).replace("-", "")
    for ts_code in ts_codes:
        rec = {"ts_code": normalize_ts_code(ts_code)}
        try:
            df = ak.stock_zh_a_minute(symbol=_sina_symbol(ts_code), period="1", adjust="")
            if df is not None and not df.empty:
                d = df.copy()
                time_col = "day" if "day" in d.columns else ("时间" if "时间" in d.columns else None)
                if time_col:
                    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")
                    d = d.loc[d[time_col].dt.strftime("%Y%m%d").eq(day)]
                    d = d.loc[(d[time_col].dt.strftime("%H:%M:%S") >= start_time) & (d[time_col].dt.strftime("%H:%M:%S") <= end_time)]
                rename = {"开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}
                d = d.rename(columns=rename)
                if not d.empty:
                    for col in ["open", "high", "low", "close", "volume", "amount"]:
                        if col in d.columns:
                            d[col] = pd.to_numeric(d[col], errors="coerce")
                    first_open = d.iloc[0].get("open")
                    last_close = d.iloc[-1].get("close")
                    high = d.get("high", pd.Series(dtype=float)).max()
                    low = d.get("low", pd.Series(dtype=float)).min()
                    vol = d.get("volume", pd.Series(dtype=float)).sum(min_count=1)
                    amt = d.get("amount", pd.Series(dtype=float)).sum(min_count=1)
                    vwap = pd.NA if pd.isna(vol) or float(vol) == 0 or pd.isna(amt) else float(amt) / float(vol)
                    rec.update({
                        "minute_bar_count_30m": int(len(d)),
                        "minute_last30_return": pd.NA if pd.isna(first_open) or float(first_open) == 0 or pd.isna(last_close) else float(last_close) / float(first_open) - 1,
                        "minute_range_pos_30m": pd.NA if pd.isna(high) or pd.isna(low) or float(high) == float(low) or pd.isna(last_close) else (float(last_close) - float(low)) / (float(high) - float(low)),
                        "minute_vwap_gap_30m": pd.NA if pd.isna(vwap) or float(vwap) == 0 or pd.isna(last_close) else float(last_close) / float(vwap) - 1,
                        "minute_vol_30m": vol,
                        "minute_amount_30m": amt,
                    })
        except Exception as exc:
            rec["minute_error"] = f"{type(exc).__name__}:{exc}"
        rows.append(rec)
        time.sleep(0.08)
    return pd.DataFrame(rows)


def fetch_lhb_features(ts_codes: list[str], trade_date: str) -> pd.DataFrame:
    base = _ensure_ts_rows(ts_codes)
    try:
        import akshare as ak
        date = str(trade_date).replace("-", "")
        df = ak.stock_lhb_detail_em(start_date=date, end_date=date)
        if df is None or df.empty:
            base["lhb_flag_1d"] = 0
            return base
        d = df.copy()
        code_col = "代码" if "代码" in d.columns else ("股票代码" if "股票代码" in d.columns else None)
        if not code_col:
            base["lhb_flag_1d"] = 0
            return base
        d["ts_code"] = d[code_col].astype(str).str.zfill(6).map(normalize_ts_code)
        grouped = d.groupby("ts_code", as_index=False).agg(lhb_count_1d=(code_col, "count"))
        for src, dst in [("净买额", "lhb_net_buy_amt"), ("买入额", "lhb_buy_amt"), ("卖出额", "lhb_sell_amt"), ("成交额", "lhb_turnover_amt")]:
            if src in d.columns:
                tmp = d.assign(**{src: pd.to_numeric(d[src], errors="coerce")}).groupby("ts_code", as_index=False)[src].sum().rename(columns={src: dst})
                grouped = grouped.merge(tmp, on="ts_code", how="left")
        grouped["lhb_flag_1d"] = 1
        return base.merge(grouped, on="ts_code", how="left").fillna({"lhb_flag_1d": 0, "lhb_count_1d": 0})
    except Exception as exc:
        base["lhb_error"] = f"{type(exc).__name__}:{exc}"
        return base


def fetch_institution_seat_features(ts_codes: list[str], trade_date: str) -> pd.DataFrame:
    base = _ensure_ts_rows(ts_codes)
    try:
        import akshare as ak
        date = str(trade_date).replace("-", "")
        df = ak.stock_lhb_jgmmtj_em(start_date=date, end_date=date)
        if df is None or df.empty:
            base["institution_lhb_flag_1d"] = 0
            return base
        d = df.copy()
        code_col = "代码" if "代码" in d.columns else ("股票代码" if "股票代码" in d.columns else None)
        if not code_col:
            base["institution_lhb_flag_1d"] = 0
            return base
        d["ts_code"] = d[code_col].astype(str).str.zfill(6).map(normalize_ts_code)
        grouped = d.groupby("ts_code", as_index=False).agg(institution_lhb_count_1d=(code_col, "count"))
        for src, dst in [("机构买入额", "institution_buy_amt"), ("机构卖出额", "institution_sell_amt"), ("机构净买额", "institution_net_buy_amt")]:
            if src in d.columns:
                tmp = d.assign(**{src: pd.to_numeric(d[src], errors="coerce")}).groupby("ts_code", as_index=False)[src].sum().rename(columns={src: dst})
                grouped = grouped.merge(tmp, on="ts_code", how="left")
        if "institution_net_buy_amt" not in grouped.columns and {"institution_buy_amt", "institution_sell_amt"}.issubset(grouped.columns):
            grouped["institution_net_buy_amt"] = grouped["institution_buy_amt"] - grouped["institution_sell_amt"]
        grouped["institution_lhb_flag_1d"] = 1
        return base.merge(grouped, on="ts_code", how="left").fillna({"institution_lhb_flag_1d": 0, "institution_lhb_count_1d": 0})
    except Exception as exc:
        base["institution_error"] = f"{type(exc).__name__}:{exc}"
        return base


def fetch_block_trade_features(ts_codes: list[str], trade_date: str) -> pd.DataFrame:
    base = _ensure_ts_rows(ts_codes)
    try:
        import akshare as ak
        date = str(trade_date).replace("-", "")
        df = ak.stock_dzjy_mrtj(start_date=date, end_date=date)
        if df is None or df.empty:
            base["block_trade_flag_1d"] = 0
            return base
        d = df.copy()
        code_col = "证券代码" if "证券代码" in d.columns else ("代码" if "代码" in d.columns else None)
        if not code_col:
            base["block_trade_flag_1d"] = 0
            return base
        d["ts_code"] = d[code_col].astype(str).str.zfill(6).map(normalize_ts_code)
        grouped = d.groupby("ts_code", as_index=False).agg(block_trade_count_1d=(code_col, "count"))
        for src, dst in [("成交额", "block_trade_amount"), ("折溢率", "block_trade_discount_rate")]:
            if src in d.columns:
                func = "sum" if src == "成交额" else "mean"
                tmp = d.assign(**{src: pd.to_numeric(d[src], errors="coerce")}).groupby("ts_code", as_index=False)[src].agg(func).rename(columns={src: dst})
                grouped = grouped.merge(tmp, on="ts_code", how="left")
        grouped["block_trade_flag_1d"] = 1
        return base.merge(grouped, on="ts_code", how="left").fillna({"block_trade_flag_1d": 0, "block_trade_count_1d": 0})
    except Exception as exc:
        base["block_trade_error"] = f"{type(exc).__name__}:{exc}"
        return base


def fetch_research_report_features(ts_codes: list[str]) -> pd.DataFrame:
    rows = []
    try:
        import akshare as ak
    except Exception as exc:
        out = _ensure_ts_rows(ts_codes)
        out["research_error"] = f"akshare_missing:{exc}"
        return out
    for ts_code in ts_codes:
        rec = {"ts_code": normalize_ts_code(ts_code)}
        try:
            df = ak.stock_research_report_em(symbol=_bare(ts_code))
            if df is not None and not df.empty:
                rec["research_report_count_30d"] = int(len(df.head(30)))
                for col in ["评级", "报告名称", "机构", "研报名称"]:
                    if col in df.columns:
                        rec[f"research_latest_{col}"] = ";".join(df[col].dropna().astype(str).head(3).tolist())[:300]
        except Exception as exc:
            rec["research_error"] = f"{type(exc).__name__}:{exc}"
        rows.append(rec)
        time.sleep(0.08)
    return pd.DataFrame(rows)


def fetch_business_features(ts_codes: list[str]) -> pd.DataFrame:
    rows = []
    try:
        import akshare as ak
    except Exception as exc:
        out = _ensure_ts_rows(ts_codes)
        out["business_error"] = f"akshare_missing:{exc}"
        return out
    for ts_code in ts_codes:
        rec = {"ts_code": normalize_ts_code(ts_code)}
        try:
            df = ak.stock_zyjs_ths(symbol=_bare(ts_code))
            if df is not None and not df.empty:
                text = " ".join(str(x) for x in df.astype(str).values.flatten().tolist()[:80])
                rec["business_text"] = text[:800]
                rec["business_keyword_count"] = int(len(set([w for w in text.replace("，", " ").replace("、", " ").split() if len(w) >= 2])))
        except Exception as exc:
            rec["business_error"] = f"{type(exc).__name__}:{exc}"
        rows.append(rec)
        time.sleep(0.08)
    return pd.DataFrame(rows)


def build_ashare_enrichment_features(
    ts_codes: list[str],
    trade_date: str,
    include_research: bool = True,
    include_business: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    normalized = [normalize_ts_code(x) for x in ts_codes]
    started = time.time()
    parts = [
        fetch_fund_flow_features(normalized),
        fetch_akshare_minute_features(normalized, trade_date),
        fetch_lhb_features(normalized, trade_date),
        fetch_institution_seat_features(normalized, trade_date),
        fetch_block_trade_features(normalized, trade_date),
    ]
    if include_research:
        parts.append(fetch_research_report_features(normalized))
    if include_business:
        parts.append(fetch_business_features(normalized))
    out = _ensure_ts_rows(normalized)
    for part in parts:
        if part is not None and not part.empty and "ts_code" in part.columns:
            out = out.merge(part.drop_duplicates("ts_code"), on="ts_code", how="left")
    out = add_ashare_enrichment_scores(out)
    manifest = {
        "trade_date": trade_date,
        "candidate_count": len(normalized),
        "elapsed_seconds": round(time.time() - started, 3),
        "columns": list(out.columns),
    }
    return out, manifest


def add_ashare_enrichment_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    bonus = pd.Series(0.0, index=out.index)
    risk = pd.Series(0.0, index=out.index)

    def rank_bonus(col: str, weight: float, ascending: bool = True):
        nonlocal bonus
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            bonus = bonus + weight * s.rank(pct=True, ascending=ascending).fillna(0.5)

    rank_bonus("fund_main_net_inflow_1d", 0.025, True)
    rank_bonus("fund_main_net_inflow_3d_sum", 0.020, True)
    rank_bonus("fund_super_net_inflow_1d", 0.015, True)
    rank_bonus("minute_last30_return", 0.020, True)
    rank_bonus("minute_range_pos_30m", 0.015, True)
    rank_bonus("minute_vwap_gap_30m", 0.015, True)
    rank_bonus("institution_net_buy_amt", 0.020, True)
    rank_bonus("lhb_net_buy_amt", 0.012, True)
    rank_bonus("research_report_count_30d", 0.006, True)

    if "block_trade_discount_rate" in out.columns:
        discount = pd.to_numeric(out["block_trade_discount_rate"], errors="coerce")
        risk = risk + discount.lt(-5).fillna(False).astype(float) * 0.04
        bonus = bonus + discount.gt(0).fillna(False).astype(float) * 0.006
    if "fund_small_net_inflow_ratio_1d" in out.columns:
        small = pd.to_numeric(out["fund_small_net_inflow_ratio_1d"], errors="coerce")
        risk = risk + small.rank(pct=True, ascending=True).fillna(0.5) * 0.008

    out["ashare_enrichment_bonus_score"] = bonus.clip(0.0, 0.12)
    out["ashare_enrichment_risk_penalty"] = risk.clip(0.0, 0.08)
    return out
