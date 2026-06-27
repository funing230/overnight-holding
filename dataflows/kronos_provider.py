# -*- coding: utf-8 -*-
"""Kronos price-prediction provider for overnight live context.

Kronos (shiyu-coder/Kronos, AAAI 2026) is a decoder-only foundation model
pre-trained on K-line sequences from 45+ global exchanges.  It predicts
future OHLCV bars from historical ones.

Integration: run offline (before market open) to pre-compute predicted returns,
then consume the CSV during the live 14:30-14:57 pipeline as an additional
scoring factor.  Do NOT call this module inside the live window — GPU inference
is too slow for 50-stock batches.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

_KRONOS_ROOT = str(Path(__file__).resolve().parents[1] / ".venv-kronos")
if _KRONOS_ROOT not in sys.path:
    # Allow running from a venv with Kronos deps
    pass


@dataclass
class KronosPredictionResult:
    """Per-stock prediction output."""
    ts_code: str
    trade_date: str
    last_close: float
    kronos_pred_return_1d: Optional[float] = None   # predicted next-day return
    kronos_pred_return_3d: Optional[float] = None   # predicted 3-day cumulative return
    kronos_pred_direction: int = 0                   # 1=up, -1=down, 0=flat
    kronos_pred_volatility_1d: Optional[float] = None
    degraded: bool = False
    degrade_reason: str = ""


@dataclass
class KronosBatchResult:
    features: list[KronosPredictionResult] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    degraded_count: int = 0
    elapsed_seconds: float = 0.0


def _import_kronos():
    """Lazy-import Kronos model classes (requires torch, huggingface_hub, etc.)."""
    kronos_src = "/tmp/Kronos"
    if kronos_src not in sys.path:
        sys.path.insert(0, kronos_src)
    from model import Kronos, KronosTokenizer, KronosPredictor
    return Kronos, KronosTokenizer, KronosPredictor


# Module-level cache — load model once per process
_model_cache: dict[str, Any] = {}


def load_kronos_model(
    model_name: str = "NeoQuasar/Kronos-small",
    tokenizer_name: str = "NeoQuasar/Kronos-Tokenizer-base",
    device: str = "cuda",
    max_context: int = 512,
):
    """Load Kronos model + tokenizer (cached)."""
    cache_key = f"{model_name}|{tokenizer_name}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    Kronos, KronosTokenizer, KronosPredictor = _import_kronos()
    tokenizer = KronosTokenizer.from_pretrained(tokenizer_name)
    model = Kronos.from_pretrained(model_name)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    _model_cache[cache_key] = predictor
    return predictor


def fetch_kronos_prediction(
    ts_code: str,
    trade_date: str,
    predictor=None,
    lookback: int = 200,
    pred_len: int = 5,
) -> KronosPredictionResult:
    """Predict next N days of K-line for a single A-share stock.

    Uses AkShare for historical data (same source as overnight_holding).
    """
    import akshare as ak

    result = KronosPredictionResult(ts_code=ts_code, trade_date=trade_date)

    # 1. Fetch historical daily data
    try:
        code = ts_code.split(".")[0]  # 000001.SZ → 000001
        df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="")
    except Exception as e:
        result.degraded = True
        result.degrade_reason = f"akshare_fetch_failed: {e}"
        return result

    if df is None or df.empty:
        result.degraded = True
        result.degrade_reason = "akshare_empty"
        return result

    # 2. Clean & normalize
    df.rename(columns={
        "日期": "date", "开盘": "open", "收盘": "close",
        "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount",
    }, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    df = df.dropna(subset=["open", "close", "high", "low"])

    if len(df) < lookback:
        result.degraded = True
        result.degrade_reason = f"insufficient_history: {len(df)} < {lookback}"
        return result

    # 3. Prepare input
    x_df = df.iloc[-lookback:][["open", "high", "low", "close", "volume", "amount"]]
    x_ts = df.iloc[-lookback:]["date"]
    last_date = df["date"].iloc[-1]
    last_close = float(df["close"].iloc[-1])
    result.last_close = last_close

    y_ts = pd.Series(pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=pred_len
    ))

    # 4. Predict
    try:
        pred = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_len, T=1.0, top_p=0.9, sample_count=3,
        )
    except Exception as e:
        result.degraded = True
        result.degrade_reason = f"prediction_failed: {e}"
        return result

    # 5. Extract features
    if pred is not None and len(pred) >= 1:
        pred_close_d1 = float(pred.iloc[0]["close"]) if len(pred) >= 1 else last_close
        pred_close_d3 = float(pred.iloc[min(2, len(pred)-1)]["close"]) if len(pred) >= 1 else last_close

        result.kronos_pred_return_1d = round((pred_close_d1 - last_close) / last_close, 6)
        result.kronos_pred_return_3d = round((pred_close_d3 - last_close) / last_close, 6)

        pred_returns_1d = []
        prev_c = last_close
        for i in range(min(pred_len, len(pred))):
            c = float(pred.iloc[i]["close"])
            pred_returns_1d.append((c - prev_c) / prev_c)
            prev_c = c
        if pred_returns_1d:
            result.kronos_pred_volatility_1d = round(
                float(pd.Series(pred_returns_1d).std()), 6
            )

        if result.kronos_pred_return_1d is not None:
            if result.kronos_pred_return_1d > 0.005:
                result.kronos_pred_direction = 1
            elif result.kronos_pred_return_1d < -0.005:
                result.kronos_pred_direction = -1
            else:
                result.kronos_pred_direction = 0

    return result


def build_kronos_batch_features(
    ts_codes: list[str],
    trade_date: str,
    model_name: str = "NeoQuasar/Kronos-small",
    lookback: int = 200,
    pred_len: int = 5,
) -> KronosBatchResult:
    """Run Kronos predictions for a batch of A-share stocks.

    Call this OFF-LINE (before market open).  It loads the model once,
    then iterates over stocks sequentially.

    Returns a KronosBatchResult with features list and per-feature DataFrames.
    """
    started = time.time()
    import torch

    result = KronosBatchResult()

    # Load model once
    try:
        predictor = load_kronos_model(model_name=model_name)
    except Exception as e:
        result.summary = {
            "status": "model_load_failed",
            "error": str(e),
            "trade_date": trade_date,
            "ts_code_count": len(ts_codes),
            "elapsed_seconds": round(time.time() - started, 1),
        }
        return result

    degraded = 0
    for code in ts_codes:
        pred = fetch_kronos_prediction(
            ts_code=code,
            trade_date=trade_date,
            predictor=predictor,
            lookback=lookback,
            pred_len=pred_len,
        )
        result.features.append(pred)
        if pred.degraded:
            degraded += 1

    result.degraded_count = degraded
    result.summary = {
        "status": "completed",
        "model": model_name,
        "trade_date": trade_date,
        "ts_code_count": len(ts_codes),
        "degraded_count": degraded,
        "success_rate": round((len(ts_codes) - degraded) / max(len(ts_codes), 1), 3),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    result.elapsed_seconds = round(time.time() - started, 1)

    return result


def kronos_features_to_dataframe(result: KronosBatchResult) -> pd.DataFrame:
    """Convert batch result to a DataFrame suitable for merging into live features."""
    rows = []
    for f in result.features:
        rows.append({
            "ts_code": f.ts_code,
            "trade_date": f.trade_date,
            "last_close": f.last_close,
            "kronos_pred_return_1d": f.kronos_pred_return_1d,
            "kronos_pred_return_3d": f.kronos_pred_return_3d,
            "kronos_pred_direction": f.kronos_pred_direction,
            "kronos_pred_volatility_1d": f.kronos_pred_volatility_1d,
            "kronos_degraded": f.degraded,
            "kronos_degrade_reason": f.degrade_reason,
        })
    return pd.DataFrame(rows)


def write_kronos_features(
    result: KronosBatchResult,
    out_csv: str | Path,
):
    """Write Kronos batch features to CSV."""
    df = kronos_features_to_dataframe(result)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out
