"""
Performance feedback tracker — Vibe-Trading inspired closed-loop learning.

Records daily Top5 predictions, verifies next-day actual performance,
and generates a compact context block for injection into the Selector prompt.

Data model:
  predictions/{trade_date}.csv  — today's Top5 snapshot (saved at end of run)
  performance_history.csv       — cumulative record: prediction → actual outcome
"""
from __future__ import annotations

import json
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_PERF_DIR = Path("data/performance")
PREDICTIONS_SUBDIR = "predictions"
HISTORY_FILE = "performance_history.csv"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_predictions(
    trade_date: str,
    final_selected_csv: str | Path,
    output_dir: str | Path = DEFAULT_PERF_DIR,
) -> Path:
    """Save today's Top5 predictions for tomorrow's verification.

    Reads the final selected CSV from a run, extracts ts_code/rank/score/price,
    and writes a predictions file keyed by trade_date.
    """
    output_dir = Path(output_dir)
    _ensure_dir(output_dir / PREDICTIONS_SUBDIR)

    df = pd.read_csv(final_selected_csv)
    cols_needed = ["ts_code", "name", "final_live_score", "last_price", "pre_close"]
    available = [c for c in cols_needed if c in df.columns]
    if "name" not in available:
        name_col = next((c for c in df.columns if "name" in c.lower()), None)
        if name_col:
            df["name"] = df[name_col]
            available.append("name")

    pred = df[available].copy()
    pred["rank"] = range(1, len(pred) + 1)
    pred["trade_date"] = trade_date

    out = output_dir / PREDICTIONS_SUBDIR / f"{trade_date}.csv"
    pred.to_csv(out, index=False)
    return out


# ── Tencent K-line helper (safe sequential, per skill guidance) ──

def _fetch_next_day_kline(ts_codes: list[str], base_date: str) -> pd.DataFrame:
    """Fetch the next trading day's OHLCV for given codes via Tencent K-line.

    Safe sequential fetching with 0.5s delay to avoid WAF.
    Returns DataFrame with ts_code, date, open, close, high, low.
    """
    rows = []
    base_dt = datetime.strptime(base_date, "%Y%m%d")
    target_date = (base_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    for code in ts_codes:
        bare = code.split(".", 1)[0]
        prefix = "sh" if bare.startswith(("6", "9", "5")) else "sz"
        key = f"{prefix}{bare}"
        url = (
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
            f"?param={key},day,,,5,qfq"
        )
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Referer": "https://gu.qq.com/",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read())
            raw = data.get("data", {})
            if isinstance(raw, dict):
                klines = raw.get(key, {}).get("qfqday", [])
            elif isinstance(raw, list):
                klines = raw[0].get("qfqday", []) if raw else []
            else:
                klines = []
            # Find the target date or closest next day
            for k in klines:
                if k[0] >= target_date:
                    rows.append({
                        "ts_code": code,
                        "date": k[0],
                        "open": float(k[1]),
                        "close": float(k[2]),
                        "high": float(k[3]),
                        "low": float(k[4]),
                    })
                    break
        except Exception:
            pass
        time.sleep(0.5)  # WAF-safe delay

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── Verification ──────────────────────────────────────────────────

def verify_predictions(
    prediction_date: str,
    data_dir: str | Path = DEFAULT_PERF_DIR,
) -> pd.DataFrame | None:
    """Verify yesterday's predictions against actual next-day prices.

    Reads predictions/{prediction_date}.csv, fetches next-day K-line data,
    computes per-stock metrics and appends to performance_history.csv.

    Returns the computed performance DataFrame, or None if no predictions found.
    """
    data_dir = Path(data_dir)
    pred_path = data_dir / PREDICTIONS_SUBDIR / f"{prediction_date}.csv"
    if not pred_path.exists():
        print(f"No predictions found for {prediction_date}")
        return None

    pred = pd.read_csv(pred_path)
    ts_codes = pred["ts_code"].tolist()
    actual = _fetch_next_day_kline(ts_codes, prediction_date)

    if actual.empty:
        print(f"No next-day price data for {prediction_date}")
        return None

    merged = pred.merge(actual, on="ts_code", how="left")
    # Drop rows with missing actual data
    merged = merged.dropna(subset=["open"])

    merged["prediction_date"] = prediction_date
    merged["actual_date"] = merged["date"]
    merged["predicted_price"] = merged["last_price"]
    merged["actual_open"] = merged["open"]
    merged["actual_close"] = merged["close"]
    merged["actual_high"] = merged["high"]
    merged["actual_low"] = merged["low"]

    # Metrics
    merged["gap_open_return"] = (
        (merged["actual_open"] - merged["predicted_price"])
        / merged["predicted_price"]
    )
    merged["intraday_return"] = (
        (merged["actual_close"] - merged["actual_open"])
        / merged["actual_open"]
    )
    merged["total_return"] = (
        (merged["actual_close"] - merged["predicted_price"])
        / merged["predicted_price"]
    )
    merged["direction_correct"] = merged["total_return"] > 0
    # Simulated 3% take-profit / 2% stop-loss based on actual high/low vs predicted
    merged["hit_take_profit"] = (
        (merged["actual_high"] - merged["predicted_price"])
        / merged["predicted_price"]
    ) >= 0.03
    merged["hit_stop_loss"] = (
        (merged["actual_low"] - merged["predicted_price"])
        / merged["predicted_price"]
    ) <= -0.02

    keep_cols = [
        "prediction_date", "rank", "ts_code", "name", "final_live_score",
        "predicted_price", "actual_date", "actual_open", "actual_close",
        "actual_high", "actual_low", "gap_open_return", "intraday_return",
        "total_return", "direction_correct", "hit_take_profit", "hit_stop_loss",
    ]
    result = merged[keep_cols].copy()

    # Append to history
    hist_path = data_dir / HISTORY_FILE
    if hist_path.exists():
        existing = pd.read_csv(hist_path)
        existing = existing[existing["prediction_date"] != prediction_date]
        combined = pd.concat([existing, result], ignore_index=True)
    else:
        combined = result
    _ensure_dir(data_dir)
    combined.to_csv(hist_path, index=False)

    return result


# ── Context builder for Selector prompt injection ──────────────────

def build_performance_context(
    trade_date: str,
    data_dir: str | Path = DEFAULT_PERF_DIR,
    lookback_days: int = 5,
) -> str:
    """Build a compact performance summary for Selector prompt injection.

    Reads recent history and formats stats like:
      "最近5个交易日 Top5方向正确率: 3/5 (60%)
       最近5日平均收益: +0.42%
       昨日Top1: 600519.SH 次日+2.3% ✓
       ..."

    Args:
        trade_date: today's date (YYYYMMDD), used to filter history up to
        data_dir: directory containing performance_history.csv
        lookback_days: number of recent prediction dates to include

    Returns:
        Markdown string ready for Selector prompt injection, or empty string.
    """
    data_dir = Path(data_dir)
    hist_path = data_dir / HISTORY_FILE
    if not hist_path.exists():
        return ""

    hist = pd.read_csv(hist_path)
    # Normalize prediction_date to YYYYMMDD string for comparison
    hist["prediction_date"] = hist["prediction_date"].astype(str).str.replace("-", "").str[:8]
    # Only use predictions before today
    hist = hist[hist["prediction_date"] < trade_date]

    if hist.empty:
        return ""

    recent_dates = sorted(hist["prediction_date"].unique(), reverse=True)[:lookback_days]
    recent = hist[hist["prediction_date"].isin(recent_dates)]

    if recent.empty:
        return ""

    total = len(recent)
    correct = int(recent["direction_correct"].sum())
    avg_return = recent["total_return"].mean()
    tp_count = int(recent["hit_take_profit"].sum())
    sl_count = int(recent["hit_stop_loss"].sum())

    lines = [
        "## 📈 历史表现回顾（自动注入）",
        "",
        f"- 最近 {len(recent_dates)} 个交易日 Top5 方向正确率: **{correct}/{total}** ({correct/total*100:.0f}%)",
        f"- 最近 {len(recent_dates)} 日等权平均收益: **{avg_return:+.2%}**",
        f"- 触发止盈(+3%): {tp_count} 次 | 触发止损(-2%): {sl_count} 次",
        "",
    ]

    # Yesterday's detail
    yesterday_data = recent[recent["prediction_date"] == recent_dates[0]].sort_values("rank")
    if not yesterday_data.empty:
        lines.append(f"**昨日 ({recent_dates[0]}) Top5 逐只结果：**")
        for _, r in yesterday_data.iterrows():
            icon = "✅" if r["direction_correct"] else "❌"
            lines.append(
                f"  - {icon} #{int(r['rank'])} {r['ts_code']} {r['name']}: "
                f"收盘{r['actual_close']:.2f} — {r['total_return']:+.2%}"
            )
        lines.append("")

    lines.append("请参考以上历史表现调整今日筛选策略。")

    return "\n".join(lines)


def load_history(
    data_dir: str | Path = DEFAULT_PERF_DIR,
) -> pd.DataFrame:
    """Load the full performance history as a DataFrame."""
    hist_path = Path(data_dir) / HISTORY_FILE
    if not hist_path.exists():
        return pd.DataFrame()
    return pd.read_csv(hist_path)
