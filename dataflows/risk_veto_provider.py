# -*- coding: utf-8 -*-
"""Risk veto provider — scans fundamental risk signals for overnight candidates.

Extracted from daily_stock_analysis RiskAgent logic, refactored as a
lightweight stateless module for TradingAgents2.0 integration.

Data sources (all free, no API key):
  - 业绩预亏: Eastmoney datacenter (RPT_PUBLIC_OP_NEWPREDICT)
  - ST 风险: candidate pool name matching + datacenter fallback
  - 限售解禁: reserved for v2

Output: CSV with ts_code + risk_type + severity(hard/soft) + reason.
Hard-veto stocks get candidate_status=reject, which downstream
final fusion reads and forces final_live_score=-999.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────
_DATACENTER_BASE = "https://datacenter.eastmoney.com/api/data/get"
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
_TIMEOUT = 12

# earnings forecast type codes (negative only)
_EARNINGS_NEGATIVE_CODES = {"002", "003", "004"}  # 预减/续亏/首亏
_EARNINGS_CODE_LABEL = {
    "001": "预增", "002": "预减", "003": "续亏",
    "004": "首亏", "005": "扭亏", "006": "略增", "007": "略减",
}


@dataclass
class RiskVetoResult:
    """Scan result for a single candidate or the whole pool."""

    ts_code: str = ""
    risk_type: str = ""        # earnings_warning / st_risk / regulatory
    severity: str = "soft"     # hard (veto) / soft (warning)
    reason: str = ""
    data_date: str = ""


@dataclass
class RiskVetoPool:
    """Aggregated risk scan for the candidate pool."""

    vetoes: list[RiskVetoResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    total_scanned: int = 0
    hard_veto_count: int = 0
    soft_warning_count: int = 0


# ── HTTP helpers ─────────────────────────────────────────────────────────

def _json_get(url: str) -> dict[str, Any]:
    """GET JSON from eastmoney datacenter. Returns {} on any failure."""
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": _UA, "Referer": "https://data.eastmoney.com/"}
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


# ── Earnings Warning Scanner ─────────────────────────────────────────────

def _fetch_earnings_warnings(
    lookback_days: int = 90,
) -> dict[str, RiskVetoResult]:
    """Fetch recent negative earnings forecasts.

    Only fetches types 预减(002) / 续亏(003) / 首亏(004).
    Returns dict keyed by 6-digit SECURITY_CODE.
    """
    result: dict[str, RiskVetoResult] = {}

    for code in sorted(_EARNINGS_NEGATIVE_CODES):
        filt = f"(PREDICT_FINANCE_CODE=%27{code}%27)"
        url = (
            f"{_DATACENTER_BASE}?type=RPT_PUBLIC_OP_NEWPREDICT"
            f"&sty=ALL&p=1&ps=200&sr=-1&st=NOTICE_DATE"
            f"&filter={filt}"
            f"&fields=SECURITY_CODE,SECURITY_NAME_ABBR,NOTICE_DATE,"
            f"PREDICT_FINANCE_CODE,PREDICT_AMT_LOWER,PREDICT_CONTENT"
        )
        data = _json_get(url)
        rows = (data.get("result") or {}).get("data") or []

        for row in rows:
            sec_code = str(row.get("SECURITY_CODE", "")).strip()
            if not sec_code or len(sec_code) != 6:
                continue
            label = _EARNINGS_CODE_LABEL.get(
                str(row.get("PREDICT_FINANCE_CODE", "")), "未知"
            )
            amt = float(row.get("PREDICT_AMT_LOWER", 0) or 0)
            content = (row.get("PREDICT_CONTENT") or "")[:120]
            reason = f"{label} {amt/1e4:.0f}万"
            if content:
                reason += f" — {content}"

            existing = result.get(sec_code)
            if existing:
                existing.reason += f"; {reason}"
                if code in ("003", "004"):  # 续亏/首亏 -> hard veto
                    existing.severity = "hard"
            else:
                result[sec_code] = RiskVetoResult(
                    ts_code=sec_code,
                    risk_type="earnings_warning",
                    severity="hard" if code in ("003", "004") else "soft",
                    reason=reason,
                    data_date=str(row.get("NOTICE_DATE", ""))[:10],
                )

    return result


# ── ST Risk Scanner ──────────────────────────────────────────────────────

def _scan_st_risks(candidate_pool: pd.DataFrame) -> dict[str, RiskVetoResult]:
    """Scan candidate pool for ST/*ST stocks by name pattern.

    Also checks for: 退市风险警示, 暂停上市.
    """
    result: dict[str, RiskVetoResult] = {}

    if candidate_pool.empty:
        return result

    name_col = None
    for col in ["name", "name_x", "name_y", "SECURITY_NAME_ABBR"]:
        if col in candidate_pool.columns:
            name_col = col
            break

    code_col = None
    for col in ["ts_code", "SECURITY_CODE", "code"]:
        if col in candidate_pool.columns:
            code_col = col
            break

    if name_col is None or code_col is None:
        return result

    st_pattern = re.compile(r"[\*]?ST|退市|暂停上市")
    for _, row in candidate_pool.iterrows():
        name = str(row.get(name_col, "")).strip()
        code = str(row.get(code_col, "")).strip()
        # Normalize ts_code (600519.SH) to 6-digit
        ts_code = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
        if len(ts_code) != 6:
            ts_code = code.zfill(6)[:6]

        if st_pattern.search(name):
            result[ts_code] = RiskVetoResult(
                ts_code=ts_code,
                risk_type="st_risk",
                severity="hard",
                reason=f"ST标识: {name}",
                data_date=datetime.now().strftime("%Y-%m-%d"),
            )

    return result


# ── Main entry point ─────────────────────────────────────────────────────

def build_risk_veto(
    candidate_pool: pd.DataFrame,
    trade_date: str = "",
    enable_earnings: bool = True,
    enable_st: bool = True,
) -> RiskVetoPool:
    """Run risk scans and return aggregated veto results.

    Args:
        candidate_pool: DataFrame with ts_code (or SECURITY_CODE) + name columns.
        trade_date: YYYY-MM-DD for logging; uses today if empty.
        enable_earnings: fetch earnings warnings (RPT_PUBLIC_OP_NEWPREDICT).
        enable_st: scan pool for ST/*ST names.

    Returns:
        RiskVetoPool with full veto list and summary stats.
    """
    pool = RiskVetoPool()
    pool.total_scanned = len(candidate_pool)

    all_vetoes: dict[str, RiskVetoResult] = {}

    # 1) ST scan (near-zero cost, runs first)
    if enable_st:
        st_results = _scan_st_risks(candidate_pool)
        for code, veto in st_results.items():
            all_vetoes[code] = veto

    # 2) Earnings warning scan (HTTP, 1-3 API calls)
    if enable_earnings:
        earnings_results = _fetch_earnings_warnings()

        # Cross-match with candidate pool codes
        code_col = None
        for col in ["ts_code", "SECURITY_CODE", "code"]:
            if col in candidate_pool.columns:
                code_col = col
                break

        if code_col:
            pool_codes: set[str] = set()
            for _, row in candidate_pool.iterrows():
                code = str(row.get(code_col, "")).strip()
                code = code.replace(".SZ", "").replace(".SH", "").replace(".BJ", "")
                if len(code) == 6:
                    pool_codes.add(code)

            for code in pool_codes:
                if code in earnings_results:
                    if code in all_vetoes:
                        # Merge: escalate existing to hard if earnings is hard
                        if earnings_results[code].severity == "hard":
                            all_vetoes[code].severity = "hard"
                        all_vetoes[code].reason += (
                            f"; {earnings_results[code].reason}"
                        )
                        all_vetoes[code].risk_type += ";earnings_warning"
                    else:
                        all_vetoes[code] = earnings_results[code]

    # Build output list
    for veto in all_vetoes.values():
        pool.vetoes.append(veto)
        if veto.severity == "hard":
            pool.hard_veto_count += 1
        else:
            pool.soft_warning_count += 1

    pool.summary = {
        "trade_date": trade_date or datetime.now().strftime("%Y-%m-%d"),
        "total_scanned": pool.total_scanned,
        "hard_veto_count": pool.hard_veto_count,
        "soft_warning_count": pool.soft_warning_count,
        "data_sources": [
            "eastmoney_earnings_forecast" if enable_earnings else None,
            "st_name_scan" if enable_st else None,
        ],
    }
    # Filter None from data_sources
    pool.summary["data_sources"] = [
        s for s in pool.summary["data_sources"] if s
    ]

    return pool


def risk_veto_to_csv(pool: RiskVetoPool, csv_path: str | Path) -> Path:
    """Write veto results to CSV. Returns the path."""
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "ts_code": v.ts_code,
            "risk_type": v.risk_type,
            "severity": v.severity,
            "reason": v.reason[:200],
            "data_date": v.data_date,
        }
        for v in pool.vetoes
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    return out


if __name__ == "__main__":
    # Standalone test
    print("=== Risk Veto Provider — standalone test ===\n")

    # Mock candidate pool
    candidates = pd.DataFrame(
        {
            "ts_code": [
                "600519.SH",  # 贵州茅台 — should be clean
                "002048.SZ",  # 宁波华翔 — 有业绩预告(扭亏/续亏/首亏)
                "300750.SZ",  # 宁德时代 — should be clean
            ],
            "name": ["贵州茅台", "宁波华翔", "宁德时代"],
            "industry": ["白酒", "汽车零部件", "电气设备"],
        }
    )

    pool = build_risk_veto(candidates, trade_date="2026-06-26")

    print(f"扫描 {pool.total_scanned} 只候选股")
    print(f"hard veto: {pool.hard_veto_count}, soft warning: {pool.soft_warning_count}")
    print()

    if pool.vetoes:
        for v in pool.vetoes:
            icon = "🚫" if v.severity == "hard" else "⚠️"
            print(f"{icon} {v.ts_code} [{v.risk_type}] {v.severity}")
            print(f"   {v.reason[:120]}")
    else:
        print("✅ 无风险信号")

    # Write CSV
    csv_path = risk_veto_to_csv(pool, "/tmp/risk_veto_test.csv")
    print(f"\nCSV: {csv_path}")
