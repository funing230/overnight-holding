from __future__ import annotations

"""Live overnight candidate inference helpers.

This module is intentionally deterministic and data-source agnostic.  The 14:55
runtime should feed it a batch market snapshot that is already collected by a
separate realtime quote adapter.  Keeping quote collection outside this module
lets the ranking path stay fast, testable, and replayable.

Expected snapshot columns (minimum):
- ts_code
- last_price

Recommended snapshot columns:
- trade_date, open, high, low, pre_close, volume, amount, pct_change, run_ts

The live scorer does not use next_open / future labels.  It reuses yesterday-and-
earlier historical features from the offline overnight feature table, then swaps
in 14:55-observable live price/range fields for today's decision.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.default_config import DEFAULT_CONFIG


THEME_LABELS = {
    "ai_terminal": "AI/终端智能化",
    "ai_infra": "AI/算力基础设施",
    "robotics": "机器人/智能制造",
    "new_energy": "新能源/储能/锂电",
    "consumer_travel": "消费/离境退税/出行",
    "macro_diplomacy": "宏观外交/风险偏好",
}

THEME_STRENGTH = {
    "news": 0.030,
    "social": 0.020,
}


@dataclass
class LiveOvernightConfig:
    history_feature_table_path: Path
    candidate_pool_size: int = 20
    top_n: int = 5
    min_price: float = 3.0
    max_abs_live_return: float = 0.095
    max_from_day_high: float = 0.08
    require_snapshot_trade_date: bool = True


LIVE_SCORE_SPECS = [
    # Historical close/overnight tendency available before today's buy decision.
    ("hist_ret_close_1d", 0.10, True),
    ("hist_ret_close_3d", 0.08, True),
    ("hist_ret_close_5d", 0.05, True),
    ("live_px_ma5_ratio", 0.08, False),
    ("live_px_ma10_ratio", 0.05, False),
    ("hist_close_range_pos_5d", 0.08, True),
    ("hist_close_drawdown_10d", 0.07, True),
    ("hist_close_vol_5d", 0.07, False),
    ("hist_overnight_prev_1d", 0.07, False),
    ("hist_overnight_prev_3d_mean", 0.07, False),
    ("hist_overnight_prev_5d_mean", 0.06, False),
    ("hist_overnight_prev_5d_std", 0.03, False),
    ("hist_overnight_positive_rate_5d", 0.05, False),
    # 14:55-observable intraday state.
    ("live_return_vs_prev_close", 0.10, True),
    ("live_range_pos", 0.07, True),
    ("from_day_high", 0.05, True),
    ("gap_days", 0.02, False),
    ("is_new_listing_180d", 0.02, False),
    ("prev_limit_move_like_1d", 0.02, True),
    ("prev_soft_outlier_1d", 0.01, True),
    # Kronos model-driven prediction (offline pre-computed)
    ("kronos_pred_return_1d", 0.06, True),
]


HIST_RENAME = {
    "ret_close_1d": "hist_ret_close_1d",
    "ret_close_3d": "hist_ret_close_3d",
    "ret_close_5d": "hist_ret_close_5d",
    "close_range_pos_5d": "hist_close_range_pos_5d",
    "close_drawdown_10d": "hist_close_drawdown_10d",
    "close_vol_5d": "hist_close_vol_5d",
    "overnight_prev_1d": "hist_overnight_prev_1d",
    "overnight_prev_3d_mean": "hist_overnight_prev_3d_mean",
    "overnight_prev_5d_mean": "hist_overnight_prev_5d_mean",
    "overnight_prev_5d_std": "hist_overnight_prev_5d_std",
    "overnight_positive_rate_5d": "hist_overnight_positive_rate_5d",
}


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(DEFAULT_CONFIG["project_dir"]).parent / p


def _as_boolish(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    return series.astype(str).str.lower().map({"true": 1.0, "false": 0.0}).fillna(0.0)


def load_history_feature_table(path: str | Path | None = None) -> pd.DataFrame:
    feature_path = _resolve_repo_path(path or DEFAULT_CONFIG["overnight_feature_table_path"])
    if not feature_path.exists():
        raise FileNotFoundError(f"Historical overnight feature table not found: {feature_path}")
    df = pd.read_csv(feature_path)
    required = {"trade_date", "ts_code", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Historical feature table missing required columns: {missing}")
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def latest_history_by_symbol(history: pd.DataFrame, trade_date: str | None = None) -> pd.DataFrame:
    hist = history.copy()
    if trade_date:
        cutoff = pd.to_datetime(trade_date, errors="coerce")
        hist = hist.loc[hist["trade_date"] < cutoff].copy()
    if hist.empty:
        raise ValueError("No historical feature rows available before requested live trade_date")
    idx = hist.groupby("ts_code")["trade_date"].idxmax()
    latest = hist.loc[idx].copy().reset_index(drop=True)
    latest = latest.rename(columns=HIST_RENAME)
    latest = latest.rename(columns={"trade_date": "history_trade_date", "close": "history_close"})
    return latest


def load_snapshot_csv(path: str | Path) -> pd.DataFrame:
    snapshot_path = Path(path)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Live snapshot CSV not found: {snapshot_path}")
    snap = pd.read_csv(snapshot_path)
    if "ts_code" not in snap.columns:
        raise ValueError("Snapshot CSV missing required column: ts_code")
    if "last_price" not in snap.columns:
        # Accept common aliases from quote vendors.
        for alias in ["price", "last", "close", "最新价"]:
            if alias in snap.columns:
                snap = snap.rename(columns={alias: "last_price"})
                break
    if "last_price" not in snap.columns:
        raise ValueError("Snapshot CSV missing required column: last_price")
    return snap.copy()


def _coalesce_snapshot_history_columns(out: pd.DataFrame) -> pd.DataFrame:
    """Tolerate enriched/recall CSVs being reused as snapshot inputs.

    Some callers accidentally pass a previously enriched recall candidate pool CSV
    back into run_live_inference() as if it were a raw realtime snapshot. After
    merging with history_latest again, columns such as history_close become
    history_close_x/history_close_y and downstream code looking for the canonical
    column name fails.

    This helper restores the canonical names by preferring the freshly merged
    history_latest columns (typically *_y when the snapshot already carried a
    stale copy), then falling back to the snapshot-side copy if needed.
    """
    canonical_cols = [
        "history_trade_date", "history_close", "name", "industry", "market", "gap_days",
        "hist_ret_close_1d", "hist_ret_close_3d", "hist_ret_close_5d",
        "close_ma5_ratio", "close_ma10_ratio", "hist_close_range_pos_5d", "hist_close_drawdown_10d",
        "hist_close_vol_5d", "hist_overnight_prev_1d", "hist_overnight_prev_3d_mean",
        "hist_overnight_prev_5d_mean", "hist_overnight_prev_5d_std",
        "hist_overnight_positive_rate_5d", "is_new_listing_180d",
        "prev_limit_move_like_1d", "prev_soft_outlier_1d",
    ]
    out = out.copy()
    for col in canonical_cols:
        if col in out.columns:
            continue
        right = f"{col}_y"
        left = f"{col}_x"
        if right in out.columns and left in out.columns:
            out[col] = out[right].combine_first(out[left])
        elif right in out.columns:
            out[col] = out[right]
        elif left in out.columns:
            out[col] = out[left]
    return out


def build_live_feature_frame(
    snapshot: pd.DataFrame,
    history_latest: pd.DataFrame,
    trade_date: str,
) -> pd.DataFrame:
    snap = snapshot.copy()
    snap["trade_date"] = str(trade_date)
    for col in ["last_price", "open", "high", "low", "pre_close", "volume", "amount", "pct_change"]:
        if col in snap.columns:
            snap[col] = pd.to_numeric(snap[col], errors="coerce")

    hist_keep = [
        c for c in [
            "ts_code", "history_trade_date", "history_close", "name", "industry", "market", "gap_days",
            "hist_ret_close_1d", "hist_ret_close_3d", "hist_ret_close_5d",
            "close_ma5_ratio", "close_ma10_ratio", "hist_close_range_pos_5d", "hist_close_drawdown_10d",
            "hist_close_vol_5d", "hist_overnight_prev_1d", "hist_overnight_prev_3d_mean",
            "hist_overnight_prev_5d_mean", "hist_overnight_prev_5d_std",
            "hist_overnight_positive_rate_5d", "is_new_listing_180d",
            "prev_limit_move_like_1d", "prev_soft_outlier_1d",
        ] if c in history_latest.columns
    ]
    out = snap.merge(history_latest[hist_keep], on="ts_code", how="inner")
    out = _coalesce_snapshot_history_columns(out)
    if out.empty:
        raise ValueError("Snapshot and historical feature table have no overlapping ts_code values")

    out["live_return_vs_prev_close"] = out["last_price"] / out["history_close"].replace(0, pd.NA) - 1.0
    if "pre_close" in out.columns:
        out["live_return_vs_pre_close"] = out["last_price"] / out["pre_close"].replace(0, pd.NA) - 1.0
    else:
        out["live_return_vs_pre_close"] = out["live_return_vs_prev_close"]

    high = out["high"] if "high" in out.columns else out["last_price"]
    low = out["low"] if "low" in out.columns else out["last_price"]
    intraday_range = (high - low).replace(0, pd.NA)
    out["live_range_pos"] = (out["last_price"] - low) / intraday_range
    out["from_day_high"] = out["last_price"] / high.replace(0, pd.NA) - 1.0
    out["from_day_low"] = out["last_price"] / low.replace(0, pd.NA) - 1.0

    # Reconstruct historical MA levels from historical close ratios when possible.
    if "close_ma5_ratio" in out.columns:
        ma5 = out["history_close"] / (1.0 + pd.to_numeric(out["close_ma5_ratio"], errors="coerce"))
        out["live_px_ma5_ratio"] = out["last_price"] / ma5.replace(0, pd.NA) - 1.0
    if "close_ma10_ratio" in out.columns:
        ma10 = out["history_close"] / (1.0 + pd.to_numeric(out["close_ma10_ratio"], errors="coerce"))
        out["live_px_ma10_ratio"] = out["last_price"] / ma10.replace(0, pd.NA) - 1.0

    out["live_near_limit_move_like"] = out["live_return_vs_pre_close"].abs() >= 0.095
    out["live_soft_outlier"] = out["live_return_vs_prev_close"].abs() >= 0.075
    out["live_intraday_pullback"] = out["from_day_high"] <= -0.05
    return out


def score_live_candidates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["overnight_live_score"] = 0.0
    for col, weight, ascending in LIVE_SCORE_SPECS:
        if col not in out.columns:
            continue
        series = _as_boolish(out[col]) if out[col].dtype == bool else pd.to_numeric(out[col], errors="coerce")
        ranks = series.rank(pct=True, ascending=ascending)
        out[f"live_score_component__{col}"] = ranks
        out["overnight_live_score"] += weight * ranks.fillna(0.5)

    penalty = pd.Series(0.0, index=out.index)
    for flag, pen in [
        ("live_near_limit_move_like", 0.20),
        ("live_soft_outlier", 0.08),
        ("live_intraday_pullback", 0.10),
        ("prev_limit_move_like_1d", 0.05),
        ("prev_soft_outlier_1d", 0.03),
    ]:
        if flag in out.columns:
            penalty += _as_boolish(out[flag]) * pen
    out["overnight_live_score"] = out["overnight_live_score"] - penalty
    out["rank_in_live_day"] = out["overnight_live_score"].rank(method="first", ascending=False)
    return out.sort_values(["rank_in_live_day", "ts_code"]).reset_index(drop=True)


def apply_live_risk_filters(df: pd.DataFrame, config: LiveOvernightConfig | None = None) -> pd.DataFrame:
    cfg = config or LiveOvernightConfig(history_feature_table_path=Path(DEFAULT_CONFIG["overnight_feature_table_path"]))
    out = df.copy()
    reasons: list[list[str]] = []
    for _, row in out.iterrows():
        r: list[str] = []
        price = row.get("last_price")
        live_ret = row.get("live_return_vs_pre_close", row.get("live_return_vs_prev_close"))
        from_high = row.get("from_day_high")
        if pd.isna(price) or float(price) < cfg.min_price:
            r.append("price_below_min_or_missing")
        if pd.notna(live_ret) and abs(float(live_ret)) > cfg.max_abs_live_return:
            r.append("near_or_beyond_daily_limit_move")
        if pd.notna(from_high) and float(from_high) < -cfg.max_from_day_high:
            r.append("large_intraday_pullback_from_high")
        if bool(row.get("live_near_limit_move_like", False)):
            r.append("live_near_limit_move_like")
        reasons.append(r)
    out["live_reject_reasons"] = [";".join(r) for r in reasons]
    out["live_pass_risk_filter"] = out["live_reject_reasons"].eq("")
    return out


def _boolish_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def _load_scorer_review_scores(review_scores_path: str | Path) -> pd.DataFrame:
    review_path = Path(review_scores_path)
    if not review_path.exists():
        raise FileNotFoundError(f"Scorer review scores not found: {review_path}")
    review = pd.read_csv(review_path)
    if "ts_code" not in review.columns:
        raise ValueError(f"Scorer review scores missing ts_code: {review_path}")
    for col, default in [
        ("agent_score", 0.5),
        ("agent_adjustment", 0.0),
        ("agent_veto", False),
        ("agent_risk_level", ""),
        ("agent_reason", ""),
    ]:
        if col not in review.columns:
            review[col] = default
    review = review[["ts_code", "agent_score", "agent_adjustment", "agent_veto", "agent_risk_level", "agent_reason"]].copy()
    review["agent_score"] = pd.to_numeric(review["agent_score"], errors="coerce").clip(0, 1).fillna(0.5)
    review["agent_adjustment"] = pd.to_numeric(review["agent_adjustment"], errors="coerce").clip(-0.2, 0.2).fillna(0.0)
    review["agent_veto"] = _boolish_series(review["agent_veto"])
    return review


def _load_selector_review_scores(selector_review_scores_path: str | Path) -> pd.DataFrame:
    review_path = Path(selector_review_scores_path)
    if not review_path.exists():
        raise FileNotFoundError(f"Selector review scores not found: {review_path}")
    review = pd.read_csv(review_path)
    if "ts_code" not in review.columns:
        raise ValueError(f"Selector review scores missing ts_code: {review_path}")
    for col, default in [
        ("heavy_score", 0.5),
        ("heavy_tier", "watch"),
        ("heavy_veto", False),
        ("heavy_adjustment", 0.0),
        ("heavy_keep_rank", pd.NA),
        ("heavy_reason", ""),
        ("heavy_risk_flags", ""),
    ]:
        if col not in review.columns:
            review[col] = default
    review = review[["ts_code", "heavy_score", "heavy_tier", "heavy_veto", "heavy_adjustment", "heavy_keep_rank", "heavy_reason", "heavy_risk_flags"]].copy()
    review["heavy_score"] = pd.to_numeric(review["heavy_score"], errors="coerce").clip(0, 1).fillna(0.5)
    review["heavy_adjustment"] = pd.to_numeric(review["heavy_adjustment"], errors="coerce").clip(-0.2, 0.2).fillna(0.0)
    review["heavy_keep_rank"] = pd.to_numeric(review["heavy_keep_rank"], errors="coerce")
    review["heavy_veto"] = _boolish_series(review["heavy_veto"])
    review["heavy_tier"] = review["heavy_tier"].astype(str).str.lower().where(
        review["heavy_tier"].astype(str).str.lower().isin(["core", "watch", "reject"]),
        "watch",
    )
    return review


def _load_social_hot_features(social_hot_features_path: str | Path) -> pd.DataFrame:
    feature_path = Path(social_hot_features_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Social hot features not found: {feature_path}")
    try:
        feat = pd.read_csv(feature_path)
    except pd.errors.EmptyDataError:
        feat = pd.DataFrame(columns=["ts_code", "hot_mention_count", "hot_source_count", "hot_best_rank", "hot_recency_hours", "social_bonus_score"])
    if feat.empty and "ts_code" not in feat.columns:
        feat = pd.DataFrame(columns=["ts_code", "hot_mention_count", "hot_source_count", "hot_best_rank", "hot_recency_hours", "social_bonus_score"])
    if "ts_code" not in feat.columns:
        raise ValueError(f"Social hot features missing ts_code: {feature_path}")
    for col, default in [
        ("hot_mention_count", 0),
        ("hot_source_count", 0),
        ("hot_best_rank", pd.NA),
        ("hot_recency_hours", pd.NA),
        ("social_bonus_score", 0.0),
    ]:
        if col not in feat.columns:
            feat[col] = default
    feat = feat[["ts_code", "hot_mention_count", "hot_source_count", "hot_best_rank", "hot_recency_hours", "social_bonus_score"]].copy()
    feat["hot_mention_count"] = pd.to_numeric(feat["hot_mention_count"], errors="coerce").fillna(0).astype(int)
    feat["hot_source_count"] = pd.to_numeric(feat["hot_source_count"], errors="coerce").fillna(0).astype(int)
    feat["hot_best_rank"] = pd.to_numeric(feat["hot_best_rank"], errors="coerce")
    feat["hot_recency_hours"] = pd.to_numeric(feat["hot_recency_hours"], errors="coerce")
    feat["social_bonus_score"] = pd.to_numeric(feat["social_bonus_score"], errors="coerce").fillna(0.0).clip(0.0, 0.05)
    return feat


def _load_theme_hot_features(theme_hot_features_path: str | Path) -> pd.DataFrame:
    feature_path = Path(theme_hot_features_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Theme hot features not found: {feature_path}")
    try:
        feat = pd.read_csv(feature_path)
    except pd.errors.EmptyDataError:
        feat = pd.DataFrame(columns=["ts_code", "theme_match_count", "theme_source_count", "theme_names", "theme_bonus_score"])
    if feat.empty and "ts_code" not in feat.columns:
        feat = pd.DataFrame(columns=["ts_code", "theme_match_count", "theme_source_count", "theme_names", "theme_bonus_score"])
    if "ts_code" not in feat.columns:
        raise ValueError(f"Theme hot features missing ts_code: {feature_path}")
    for col, default in [
        ("theme_match_count", 0),
        ("theme_source_count", 0),
        ("theme_names", ""),
        ("theme_bonus_score", 0.0),
        ("openclaw_event_strength", 0.0),
        ("openclaw_positive_signal", 0.0),
        ("openclaw_risk_penalty", 0.0),
        ("openclaw_theme_count", 0),
        ("openclaw_theme_names", ""),
        ("openclaw_macro_mentions", 0),
        ("openclaw_social_mentions", 0),
        ("openclaw_feature_score", 0.0),
        ("openclaw_risk_flags", ""),
        ("openclaw_catalyst_summary", ""),
    ]:
        if col not in feat.columns:
            feat[col] = default
    feat = feat[["ts_code", "theme_match_count", "theme_source_count", "theme_names", "theme_bonus_score"]].copy()
    feat["theme_match_count"] = pd.to_numeric(feat["theme_match_count"], errors="coerce").fillna(0).astype(int)
    feat["theme_source_count"] = pd.to_numeric(feat["theme_source_count"], errors="coerce").fillna(0).astype(int)
    feat["theme_names"] = feat["theme_names"].fillna("").astype(str)
    feat["theme_bonus_score"] = pd.to_numeric(feat["theme_bonus_score"], errors="coerce").fillna(0.0).clip(0.0, 0.08)
    return feat


def _load_openclaw_features(openclaw_features_path: str | Path) -> pd.DataFrame:
    feature_path = Path(openclaw_features_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"OpenClaw features not found: {feature_path}")
    try:
        feat = pd.read_csv(feature_path)
    except pd.errors.EmptyDataError:
        feat = pd.DataFrame(columns=["ts_code", "openclaw_event_strength", "openclaw_positive_signal", "openclaw_risk_penalty", "openclaw_theme_count", "openclaw_theme_names", "openclaw_macro_mentions", "openclaw_social_mentions", "openclaw_feature_score", "openclaw_risk_flags", "openclaw_catalyst_summary"])
    if feat.empty and "ts_code" not in feat.columns:
        feat = pd.DataFrame(columns=["ts_code", "openclaw_event_strength", "openclaw_positive_signal", "openclaw_risk_penalty", "openclaw_theme_count", "openclaw_theme_names", "openclaw_macro_mentions", "openclaw_social_mentions", "openclaw_feature_score", "openclaw_risk_flags", "openclaw_catalyst_summary"])
    if "ts_code" not in feat.columns:
        raise ValueError(f"OpenClaw features missing ts_code: {feature_path}")
    for col, default in [
        ("openclaw_event_strength", 0.0),
        ("openclaw_positive_signal", 0.0),
        ("openclaw_risk_penalty", 0.0),
        ("openclaw_theme_count", 0),
        ("openclaw_theme_names", ""),
        ("openclaw_macro_mentions", 0),
        ("openclaw_social_mentions", 0),
        ("openclaw_feature_score", 0.0),
        ("openclaw_risk_flags", ""),
        ("openclaw_catalyst_summary", ""),
    ]:
        if col not in feat.columns:
            feat[col] = default
    feat = feat[["ts_code", "openclaw_event_strength", "openclaw_positive_signal", "openclaw_risk_penalty", "openclaw_theme_count", "openclaw_theme_names", "openclaw_macro_mentions", "openclaw_social_mentions", "openclaw_feature_score", "openclaw_risk_flags", "openclaw_catalyst_summary"]].copy()
    for col in ["openclaw_event_strength", "openclaw_positive_signal", "openclaw_risk_penalty", "openclaw_feature_score"]:
        feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(0.0)
    for col in ["openclaw_theme_count", "openclaw_macro_mentions", "openclaw_social_mentions"]:
        feat[col] = pd.to_numeric(feat[col], errors="coerce").fillna(0).astype(int)
    for col in ["openclaw_theme_names", "openclaw_risk_flags", "openclaw_catalyst_summary"]:
        feat[col] = feat[col].fillna("").astype(str)
    feat["openclaw_feature_score"] = feat["openclaw_feature_score"].clip(-0.08, 0.08)
    feat["openclaw_risk_penalty"] = feat["openclaw_risk_penalty"].clip(0.0, 0.08)
    return feat


def apply_scorer_review_fusion(
    scored: pd.DataFrame,
    review_scores_path: str | Path | None = None,
    live_weight: float = 0.75,
    agent_weight: float = 0.25,
) -> pd.DataFrame:
    """Fuse deterministic live score with pre-14:55 TradingAgents review scores.

    Expected review CSV columns:
    - ts_code
    - agent_score in [0, 1]
    - agent_adjustment in [-0.2, 0.2]
    - agent_veto boolean-ish
    - agent_risk_level
    - agent_reason

    Vetoed candidates are not deleted here; they receive a very low
    final_live_score and are excluded by select_live_topn via
    live_pass_risk_filter=False.
    """
    out = scored.copy()
    out["final_live_score"] = out["overnight_live_score"]
    out["agent_score"] = pd.NA
    out["agent_adjustment"] = 0.0
    out["agent_veto"] = False
    out["agent_risk_level"] = ""
    out["agent_reason"] = ""
    if not review_scores_path:
        out["rank_in_final_live_day"] = out["final_live_score"].rank(method="first", ascending=False)
        out["rank_in_live_day"] = out["rank_in_final_live_day"]
        return out.sort_values(["rank_in_live_day", "ts_code"]).reset_index(drop=True)

    review = _load_scorer_review_scores(review_scores_path)

    out = out.merge(review, on="ts_code", how="left", suffixes=("", "_review"))
    for col in ["agent_score", "agent_adjustment", "agent_veto", "agent_risk_level", "agent_reason"]:
        review_col = f"{col}_review"
        if review_col in out.columns:
            out[col] = out[review_col].combine_first(out[col])
            out = out.drop(columns=[review_col])
    out["agent_score"] = pd.to_numeric(out["agent_score"], errors="coerce").fillna(0.5)
    out["agent_adjustment"] = pd.to_numeric(out["agent_adjustment"], errors="coerce").fillna(0.0)
    out["agent_veto"] = _boolish_series(out["agent_veto"])
    risk_penalty = out["agent_risk_level"].astype(str).str.lower().map({"low": 0.0, "medium": 0.03, "high": 0.12}).fillna(0.03)
    out["final_live_score"] = live_weight * out["overnight_live_score"] + agent_weight * out["agent_score"] + out["agent_adjustment"] - risk_penalty
    out.loc[out["agent_veto"], "final_live_score"] = -999.0
    if "live_reject_reasons" not in out.columns:
        out["live_reject_reasons"] = ""
    out.loc[out["agent_veto"], "live_reject_reasons"] = out.loc[out["agent_veto"], "live_reject_reasons"].astype(str).where(
        out.loc[out["agent_veto"], "live_reject_reasons"].astype(str).eq(""),
        out.loc[out["agent_veto"], "live_reject_reasons"].astype(str) + ";",
    ) + "agent_veto"
    if "live_pass_risk_filter" not in out.columns:
        out["live_pass_risk_filter"] = True
    out.loc[out["agent_veto"], "live_pass_risk_filter"] = False
    out["rank_in_final_live_day"] = out["final_live_score"].rank(method="first", ascending=False)
    out["rank_in_live_day"] = out["rank_in_final_live_day"]
    return out.sort_values(["rank_in_live_day", "ts_code"]).reset_index(drop=True)


def _load_ashare_enrichment_features(ashare_enrichment_features_path: str | Path) -> pd.DataFrame:
    feature_path = Path(ashare_enrichment_features_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"A-share enrichment features not found: {feature_path}")
    try:
        feat = pd.read_csv(feature_path)
    except pd.errors.EmptyDataError:
        feat = pd.DataFrame(columns=["ts_code", "ashare_enrichment_bonus_score", "ashare_enrichment_risk_penalty"])
    if feat.empty and "ts_code" not in feat.columns:
        feat = pd.DataFrame(columns=["ts_code", "ashare_enrichment_bonus_score", "ashare_enrichment_risk_penalty"])
    if "ts_code" not in feat.columns:
        raise ValueError(f"A-share enrichment features missing ts_code: {feature_path}")
    for col, default in [("ashare_enrichment_bonus_score", 0.0), ("ashare_enrichment_risk_penalty", 0.0)]:
        if col not in feat.columns:
            feat[col] = default
    keep = [c for c in feat.columns if c == "ts_code" or c.startswith(("fund_", "minute_", "lhb_", "institution_", "block_trade_", "research_", "business_", "ashare_"))]
    feat = feat[keep].copy()
    feat["ashare_enrichment_bonus_score"] = pd.to_numeric(feat["ashare_enrichment_bonus_score"], errors="coerce").fillna(0.0).clip(0.0, 0.12)
    feat["ashare_enrichment_risk_penalty"] = pd.to_numeric(feat["ashare_enrichment_risk_penalty"], errors="coerce").fillna(0.0).clip(0.0, 0.08)
    return feat.drop_duplicates("ts_code", keep="last")





def _load_kronos_features(kronos_features_path: str | Path) -> pd.DataFrame:
    """Load Kronos pre-computed prediction features."""
    feature_path = Path(kronos_features_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Kronos features not found: {feature_path}")
    try:
        feat = pd.read_csv(feature_path)
    except pd.errors.EmptyDataError:
        feat = pd.DataFrame(columns=["ts_code", "kronos_pred_return_1d", "kronos_degraded"])
    if "ts_code" not in feat.columns:
        raise ValueError(f"Kronos features missing ts_code: {feature_path}")
    for col, default in [("kronos_pred_return_1d", pd.NA), ("kronos_degraded", False)]:
        if col not in feat.columns:
            feat[col] = default
    feat = feat[["ts_code", "kronos_pred_return_1d", "kronos_degraded"]].copy()
    feat["kronos_pred_return_1d"] = pd.to_numeric(feat["kronos_pred_return_1d"], errors="coerce")
    feat["kronos_degraded"] = feat["kronos_degraded"].astype(str).str.lower().isin(["true", "1", "yes"])
    return feat.drop_duplicates("ts_code", keep="last")

def _load_dsa_scores(dsa_scores_path: str | Path) -> pd.DataFrame:
    """Load DSA Path B analysis scores."""
    p = Path(dsa_scores_path)
    if not p.exists():
        return pd.DataFrame(columns=["ts_code", "dsa_score", "dsa_operation", "dsa_available"])
    df = pd.read_csv(p)
    for col, default in [("dsa_score", 0.5), ("dsa_operation", "hold"), ("dsa_available", False)]:
        if col not in df.columns:
            df[col] = default
    return df[["ts_code", "dsa_score", "dsa_operation", "dsa_available"]]

def _load_xueqiu_hot_features(xueqiu_hot_features_path):
    feature_path = Path(xueqiu_hot_features_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Xueqiu hot features not found: {feature_path}")
    try:
        feat = pd.read_csv(feature_path)
    except pd.errors.EmptyDataError:
        feat = pd.DataFrame(columns=["ts_code", "xueqiu_bonus_score"])
    if feat.empty and "ts_code" not in feat.columns:
        feat = pd.DataFrame(columns=["ts_code", "xueqiu_bonus_score"])
    if "ts_code" not in feat.columns:
        raise ValueError(f"Xueqiu hot features missing ts_code: {feature_path}")
    if "xueqiu_bonus_score" not in feat.columns:
        feat["xueqiu_bonus_score"] = 0.0
    feat = feat[["ts_code", "xueqiu_bonus_score"]].copy()
    feat["xueqiu_bonus_score"] = pd.to_numeric(feat["xueqiu_bonus_score"], errors="coerce").fillna(0.0).clip(0.0, 0.06)
    return feat.drop_duplicates("ts_code", keep="last")


def _load_twitter_features(twitter_features_path):
    feature_path = Path(twitter_features_path)
    if not feature_path.exists():
        raise FileNotFoundError(f"Twitter features not found: {feature_path}")
    try:
        feat = pd.read_csv(feature_path)
    except pd.errors.EmptyDataError:
        feat = pd.DataFrame(columns=["ts_code", "twitter_bonus_score"])
    if feat.empty and "ts_code" not in feat.columns:
        feat = pd.DataFrame(columns=["ts_code", "twitter_bonus_score"])
    if "ts_code" not in feat.columns:
        raise ValueError(f"Twitter features missing ts_code: {feature_path}")
    if "twitter_bonus_score" not in feat.columns:
        feat["twitter_bonus_score"] = 0.0
    feat = feat[["ts_code", "twitter_bonus_score"]].copy()
    feat["twitter_bonus_score"] = pd.to_numeric(feat["twitter_bonus_score"], errors="coerce").fillna(0.0).clip(0.0, 0.04)
    return feat.drop_duplicates("ts_code", keep="last")



def apply_multi_stage_review_fusion(
    scored: pd.DataFrame,
    selector_review_scores_path: str | Path | None = None,
    scorer_review_scores_path: str | Path | None = None,
    dsa_scores_path: str | Path | None = None,
    social_hot_features_path: str | Path | None = None,
    theme_hot_features_path: str | Path | None = None,
    openclaw_features_path: str | Path | None = None,
    xueqiu_hot_features_path: str | Path | None = None,
    twitter_features_path: str | Path | None = None,
    ashare_enrichment_features_path: str | Path | None = None,
    kronos_features_path: str | Path | None = None,
    live_weight: float = 0.60,
    heavy_weight: float = 0.25,
    light_weight: float = 0.15,
    dsa_weight: float = 0.05,
) -> pd.DataFrame:
    """Fuse deterministic live score with selector Top50 review and scorer Top15 review.

    Selector review is the earlier Top50 -> Top15 research stage.  Scorer review is
    the later fast pre-close stage.  Either input may be absent; missing per-row
    scores fall back to neutral values."""
    out = scored.copy()
    out["final_live_score"] = out["overnight_live_score"]

    # Initialize all review columns so downstream summaries stay stable.
    for col, default in [
        ("heavy_score", pd.NA),
        ("heavy_tier", ""),
        ("heavy_veto", False),
        ("heavy_adjustment", 0.0),
        ("heavy_keep_rank", pd.NA),
        ("heavy_reason", ""),
        ("heavy_risk_flags", ""),
        ("agent_score", pd.NA),
        ("agent_adjustment", 0.0),
        ("agent_veto", False),
        ("agent_risk_level", ""),
        ("agent_reason", ""),
        ("hot_mention_count", 0),
        ("hot_source_count", 0),
        ("hot_best_rank", pd.NA),
        ("hot_recency_hours", pd.NA),
        ("social_bonus_score", 0.0),
        ("theme_match_count", 0),
        ("theme_source_count", 0),
        ("theme_names", ""),
        ("theme_bonus_score", 0.0),
        ("openclaw_event_strength", 0.0),
        ("openclaw_positive_signal", 0.0),
        ("openclaw_risk_penalty", 0.0),
        ("openclaw_theme_count", 0),
        ("openclaw_theme_names", ""),
        ("openclaw_macro_mentions", 0),
        ("openclaw_social_mentions", 0),
        ("openclaw_feature_score", 0.0),
        ("openclaw_risk_flags", ""),
        ("openclaw_catalyst_summary", ""),
        ("ashare_enrichment_bonus_score", 0.0),
        ("ashare_enrichment_risk_penalty", 0.0),
        ("kronos_pred_return_1d", pd.NA),
        ("kronos_degraded", False),
        ("dsa_score", 0.5),
        ("dsa_operation", "hold"),
        ("dsa_available", False),
    ]:
        out[col] = default

    if selector_review_scores_path:
        heavy = _load_selector_review_scores(selector_review_scores_path)
        out = out.merge(heavy, on="ts_code", how="left", suffixes=("", "_selector_review"))
        for col in ["heavy_score", "heavy_tier", "heavy_veto", "heavy_adjustment", "heavy_keep_rank", "heavy_reason", "heavy_risk_flags"]:
            review_col = f"{col}_selector_review"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col])
                out = out.drop(columns=[review_col])

    if scorer_review_scores_path:
        light = _load_scorer_review_scores(scorer_review_scores_path)
        out = out.merge(light, on="ts_code", how="left", suffixes=("", "_scorer_review"))
        for col in ["agent_score", "agent_adjustment", "agent_veto", "agent_risk_level", "agent_reason"]:
            review_col = f"{col}_scorer_review"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col])
                out = out.drop(columns=[review_col])

    if social_hot_features_path:
        social = _load_social_hot_features(social_hot_features_path)
        out = out.merge(social, on="ts_code", how="left", suffixes=("", "_social_hot"))
        for col in ["hot_mention_count", "hot_source_count", "hot_best_rank", "hot_recency_hours", "social_bonus_score"]:
            review_col = f"{col}_social_hot"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col])
                out = out.drop(columns=[review_col])

    if theme_hot_features_path:
        theme = _load_theme_hot_features(theme_hot_features_path)
        out = out.merge(theme, on="ts_code", how="left", suffixes=("", "_theme_hot"))
        for col in ["theme_match_count", "theme_source_count", "theme_names", "theme_bonus_score"]:
            review_col = f"{col}_theme_hot"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col])
                out = out.drop(columns=[review_col])

    if openclaw_features_path:
        openclaw = _load_openclaw_features(openclaw_features_path)
        out = out.merge(openclaw, on="ts_code", how="left", suffixes=("", "_openclaw"))
        for col in ["openclaw_event_strength", "openclaw_positive_signal", "openclaw_risk_penalty", "openclaw_theme_count", "openclaw_theme_names", "openclaw_macro_mentions", "openclaw_social_mentions", "openclaw_feature_score", "openclaw_risk_flags", "openclaw_catalyst_summary"]:
            review_col = f"{col}_openclaw"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col])
                out = out.drop(columns=[review_col])

    if ashare_enrichment_features_path:
        ashare = _load_ashare_enrichment_features(ashare_enrichment_features_path)
        out = out.merge(ashare, on="ts_code", how="left", suffixes=("", "_ashare"))
        for col in [c for c in ashare.columns if c != "ts_code"]:
            review_col = f"{col}_ashare"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col]) if col in out.columns else out[review_col]
                out = out.drop(columns=[review_col])

    if kronos_features_path:
        kronos = _load_kronos_features(kronos_features_path)
        out = out.merge(kronos, on="ts_code", how="left", suffixes=("", "_kronos"))
        for col in ["kronos_pred_return_1d", "kronos_degraded"]:
            review_col = f"{col}_kronos"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col])
                out = out.drop(columns=[review_col])

    if xueqiu_hot_features_path:
        xq = _load_xueqiu_hot_features(xueqiu_hot_features_path)
        out = out.merge(xq, on="ts_code", how="left", suffixes=("", "_xq"))
        for col in ["xueqiu_bonus_score"]:
            review_col = f"{col}_xq"
            if review_col in out.columns:
                if col in out.columns:
                    out[col] = out[review_col].combine_first(out[col])
                else:
                    out[col] = out[review_col]
                out = out.drop(columns=[review_col])

    if twitter_features_path:
        tw = _load_twitter_features(twitter_features_path)
        out = out.merge(tw, on="ts_code", how="left", suffixes=("", "_tw"))
        for col in ["twitter_bonus_score"]:
            review_col = f"{col}_tw"
            if review_col in out.columns:
                if col in out.columns:
                    out[col] = out[review_col].combine_first(out[col])
                else:
                    out[col] = out[review_col]
                out = out.drop(columns=[review_col])

    if dsa_scores_path:
        dsa = _load_dsa_scores(dsa_scores_path)
        out = out.merge(dsa, on="ts_code", how="left", suffixes=("", "_dsa"))
        for col in ["dsa_score", "dsa_operation", "dsa_available"]:
            review_col = f"{col}_dsa"
            if review_col in out.columns:
                out[col] = out[review_col].combine_first(out[col])
                out = out.drop(columns=[review_col])

    out["heavy_score"] = pd.to_numeric(out["heavy_score"], errors="coerce").fillna(0.5)
    out["heavy_adjustment"] = pd.to_numeric(out["heavy_adjustment"], errors="coerce").fillna(0.0)
    out["heavy_veto"] = _boolish_series(out["heavy_veto"])
    out["heavy_tier"] = out["heavy_tier"].astype(str).str.lower().replace({"": "watch", "<na>": "watch", "nan": "watch"})
    out["agent_score"] = pd.to_numeric(out["agent_score"], errors="coerce").fillna(0.5)
    out["agent_adjustment"] = pd.to_numeric(out["agent_adjustment"], errors="coerce").fillna(0.0)
    out["agent_veto"] = _boolish_series(out["agent_veto"])
    out["hot_mention_count"] = pd.to_numeric(out["hot_mention_count"], errors="coerce").fillna(0).astype(int)
    out["hot_source_count"] = pd.to_numeric(out["hot_source_count"], errors="coerce").fillna(0).astype(int)
    out["hot_best_rank"] = pd.to_numeric(out["hot_best_rank"], errors="coerce")
    out["hot_recency_hours"] = pd.to_numeric(out["hot_recency_hours"], errors="coerce")
    out["social_bonus_score"] = pd.to_numeric(out["social_bonus_score"], errors="coerce").fillna(0.0).clip(0.0, 0.05)
    out["theme_match_count"] = pd.to_numeric(out["theme_match_count"], errors="coerce").fillna(0).astype(int)
    out["theme_source_count"] = pd.to_numeric(out["theme_source_count"], errors="coerce").fillna(0).astype(int)
    out["theme_names"] = out["theme_names"].fillna("").astype(str)
    out["theme_bonus_score"] = pd.to_numeric(out["theme_bonus_score"], errors="coerce").fillna(0.0).clip(0.0, 0.08)
    out["openclaw_event_strength"] = pd.to_numeric(out["openclaw_event_strength"], errors="coerce").fillna(0.0)
    out["openclaw_positive_signal"] = pd.to_numeric(out["openclaw_positive_signal"], errors="coerce").fillna(0.0)
    out["openclaw_risk_penalty"] = pd.to_numeric(out["openclaw_risk_penalty"], errors="coerce").fillna(0.0).clip(0.0, 0.08)
    out["openclaw_theme_count"] = pd.to_numeric(out["openclaw_theme_count"], errors="coerce").fillna(0).astype(int)
    out["openclaw_theme_names"] = out["openclaw_theme_names"].fillna("").astype(str)
    out["openclaw_macro_mentions"] = pd.to_numeric(out["openclaw_macro_mentions"], errors="coerce").fillna(0).astype(int)
    out["openclaw_social_mentions"] = pd.to_numeric(out["openclaw_social_mentions"], errors="coerce").fillna(0).astype(int)
    out["openclaw_feature_score"] = pd.to_numeric(out["openclaw_feature_score"], errors="coerce").fillna(0.0).clip(-0.08, 0.08)
    out["openclaw_risk_flags"] = out["openclaw_risk_flags"].fillna("").astype(str)
    out["openclaw_catalyst_summary"] = out["openclaw_catalyst_summary"].fillna("").astype(str)
    out["ashare_enrichment_bonus_score"] = pd.to_numeric(out["ashare_enrichment_bonus_score"], errors="coerce").fillna(0.0).clip(0.0, 0.12)
    out["ashare_enrichment_risk_penalty"] = pd.to_numeric(out["ashare_enrichment_risk_penalty"], errors="coerce").fillna(0.0).clip(0.0, 0.08)
    out["kronos_pred_return_1d"] = pd.to_numeric(out["kronos_pred_return_1d"], errors="coerce")
    kronos_bonus = out["kronos_pred_return_1d"].fillna(0.0).clip(-0.06, 0.06)
    kronos_bonus = kronos_bonus.where(out["kronos_degraded"].fillna(True).eq(False), 0.0)

    heavy_tier_penalty = out["heavy_tier"].map({"core": 0.0, "watch": 0.05, "reject": 0.20}).fillna(0.05)
    light_risk_penalty = out["agent_risk_level"].astype(str).str.lower().map({"low": 0.0, "medium": 0.03, "high": 0.12}).fillna(0.03)
    out["final_live_score"] = (
        live_weight * out["overnight_live_score"]
        + heavy_weight * out["heavy_score"]
        + light_weight * out["agent_score"]
        + dsa_weight * out["dsa_score"]
        + out["heavy_adjustment"]
        + out["agent_adjustment"]
        + out["social_bonus_score"]
        + out["theme_bonus_score"]
        + out["openclaw_feature_score"]
        + out["ashare_enrichment_bonus_score"]
        + out.get("xueqiu_bonus_score", 0.0)
        + out.get("twitter_bonus_score", 0.0)
        + kronos_bonus
        - heavy_tier_penalty
        - light_risk_penalty
        - out["openclaw_risk_penalty"]
        - out["ashare_enrichment_risk_penalty"]
    )

    veto_mask = out["heavy_veto"] | out["agent_veto"] | out["heavy_tier"].eq("reject")
    out.loc[veto_mask, "final_live_score"] = -999.0
    if "live_reject_reasons" not in out.columns:
        out["live_reject_reasons"] = ""
    if "live_pass_risk_filter" not in out.columns:
        out["live_pass_risk_filter"] = True
    reason = pd.Series("", index=out.index)
    reason = reason.where(~out["heavy_veto"], reason + ";heavy_veto")
    reason = reason.where(~out["heavy_tier"].eq("reject"), reason + ";heavy_reject")
    reason = reason.where(~out["agent_veto"], reason + ";agent_veto")
    reason = reason.str.strip(";")
    has_reason = reason.ne("")
    out.loc[has_reason, "live_reject_reasons"] = out.loc[has_reason, "live_reject_reasons"].astype(str).where(
        out.loc[has_reason, "live_reject_reasons"].astype(str).eq(""),
        out.loc[has_reason, "live_reject_reasons"].astype(str) + ";",
    ) + reason.loc[has_reason]
    out.loc[veto_mask, "live_pass_risk_filter"] = False
    out["rank_in_final_live_day"] = out["final_live_score"].rank(method="first", ascending=False)
    out["rank_in_live_day"] = out["rank_in_final_live_day"]
    return out.sort_values(["rank_in_live_day", "ts_code"]).reset_index(drop=True)


def select_live_topn(scored: pd.DataFrame, top_n: int = 5, candidate_pool_size: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranked = scored.sort_values(["rank_in_live_day", "ts_code"]).copy()
    pool = ranked.head(candidate_pool_size).copy()
    selected = ranked.loc[ranked["live_pass_risk_filter"]].head(top_n).copy()
    return pool.reset_index(drop=True), selected.reset_index(drop=True)


def run_live_inference(
    snapshot_csv: str | Path,
    trade_date: str,
    history_feature_table_path: str | Path | None = None,
    top_n: int = 5,
    candidate_pool_size: int = 20,
    review_scores_path: str | Path | None = None,
    selector_review_scores_path: str | Path | None = None,
    scorer_review_scores_path: str | Path | None = None,
    social_hot_features_path: str | Path | None = None,
    theme_hot_features_path: str | Path | None = None,
    openclaw_features_path: str | Path | None = None,
    xueqiu_hot_features_path: str | Path | None = None,
    twitter_features_path: str | Path | None = None,
    ashare_enrichment_features_path: str | Path | None = None,
    kronos_features_path: str | Path | None = None,
    dsa_scores_path: str | Path | None = None,
    live_weight: float = 0.75,
    agent_weight: float = 0.25,
    heavy_weight: float = 0.25,
    light_weight: float = 0.15,
    dsa_weight: float = 0.05,
) -> dict[str, Any]:
    cfg = LiveOvernightConfig(
        history_feature_table_path=Path(history_feature_table_path or DEFAULT_CONFIG["overnight_feature_table_path"]),
        top_n=top_n,
        candidate_pool_size=candidate_pool_size,
    )
    history = load_history_feature_table(cfg.history_feature_table_path)
    latest = latest_history_by_symbol(history, trade_date=trade_date)
    snapshot = load_snapshot_csv(snapshot_csv)
    features = build_live_feature_frame(snapshot, latest, trade_date=trade_date)
    scored = score_live_candidates(features)
    scored = apply_live_risk_filters(scored, cfg)
    if selector_review_scores_path or scorer_review_scores_path:
        scored = apply_multi_stage_review_fusion(
            scored,
            selector_review_scores_path=selector_review_scores_path,
            scorer_review_scores_path=scorer_review_scores_path,
            social_hot_features_path=social_hot_features_path,
            theme_hot_features_path=theme_hot_features_path,
            openclaw_features_path=openclaw_features_path,
            xueqiu_hot_features_path=xueqiu_hot_features_path,
            twitter_features_path=twitter_features_path,
            ashare_enrichment_features_path=ashare_enrichment_features_path,
            kronos_features_path=kronos_features_path,
            dsa_scores_path=dsa_scores_path,
            live_weight=live_weight,
            heavy_weight=heavy_weight,
            light_weight=light_weight,
            dsa_weight=dsa_weight,
        )
    else:
        scored = apply_scorer_review_fusion(
            scored,
            review_scores_path=review_scores_path,
            live_weight=live_weight,
            agent_weight=agent_weight,
        )
    pool, selected = select_live_topn(scored, top_n=top_n, candidate_pool_size=candidate_pool_size)
    return {
        "trade_date": str(trade_date),
        "snapshot_csv": str(snapshot_csv),
        "history_feature_table_path": str(cfg.history_feature_table_path),
        "features": features,
        "scored": scored,
        "candidate_pool": pool,
        "selected": selected,
    }
