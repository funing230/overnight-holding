#!/usr/bin/env python3
"""Batch experiments for Top-5 overnight factor weights and filter rules.

Workflow:
1. Load unified feature table produced by build_overnight_feature_table.py
2. Apply one or more filter-rule variants
3. Apply one or more factor-weight variants to rescore names within each day
4. Build Top-5 candidate table for each variant
5. Reuse the baseline overnight backtest to evaluate each variant

Outputs:
  data/overnight_mvp/experiments/top5_factor_filter_batch_<stem>.csv
  data/overnight_mvp/experiments/top5_factor_filter_batch_<stem>.md
  data/overnight_mvp/experiments/inputs/<variant>.csv
"""

from __future__ import annotations

import argparse
import copy
from itertools import product
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_overnight_feature_table import build_topn_input
from backtest_overnight_topn_baseline import run_backtest


DEFAULT_FEATURES = Path("data/overnight_mvp/features/overnight_features_20260401_20260410.csv")
DEFAULT_OUTDIR = Path("data/overnight_mvp/experiments")
DEFAULT_TOP_N = 5
DEFAULT_EXIT_MODE = "open"
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
DEFAULT_FEE_BPS = 10.0
DEFAULT_SLIPPAGE_BPS = 5.0

BASE_SCORE_SPECS = [
    ("ret_close_1d", 0.22, True),
    ("ret_close_3d", 0.14, True),
    ("ret_close_5d", 0.08, True),
    ("close_ma5_ratio", 0.12, False),
    ("close_ma10_ratio", 0.10, False),
    ("close_vol_5d", 0.10, False),
    ("overnight_prev_1d", 0.10, False),
    ("overnight_prev_3d_mean", 0.08, False),
    ("gap_days", 0.03, False),
    ("is_new_listing_180d", 0.03, False),
]

WEIGHT_VARIANTS = {
    "baseline": {},
    "mean_revert_heavy": {
        "ret_close_1d": 0.28,
        "ret_close_3d": 0.18,
        "ret_close_5d": 0.10,
        "close_ma5_ratio": 0.14,
        "close_ma10_ratio": 0.10,
        "close_vol_5d": 0.06,
        "overnight_prev_1d": 0.06,
        "overnight_prev_3d_mean": 0.03,
        "gap_days": 0.03,
        "is_new_listing_180d": 0.02,
    },
    "stability_heavy": {
        "ret_close_1d": 0.16,
        "ret_close_3d": 0.12,
        "ret_close_5d": 0.06,
        "close_ma5_ratio": 0.10,
        "close_ma10_ratio": 0.08,
        "close_vol_5d": 0.22,
        "overnight_prev_1d": 0.08,
        "overnight_prev_3d_mean": 0.10,
        "gap_days": 0.03,
        "is_new_listing_180d": 0.05,
    },
    "gap_aware": {
        "ret_close_1d": 0.18,
        "ret_close_3d": 0.12,
        "ret_close_5d": 0.08,
        "close_ma5_ratio": 0.10,
        "close_ma10_ratio": 0.08,
        "close_vol_5d": 0.08,
        "overnight_prev_1d": 0.12,
        "overnight_prev_3d_mean": 0.12,
        "gap_days": 0.07,
        "is_new_listing_180d": 0.05,
    },
}

FILTER_VARIANTS = {
    "base": {
        "require_trainable": True,
        "drop_long_gap": False,
        "drop_limit_move_like": False,
        "drop_soft_outlier": False,
        "drop_extreme": False,
        "drop_new_listing_180d": False,
        "max_gap_days": None,
        "max_close_vol_5d": None,
    },
    "strict_risk": {
        "require_trainable": True,
        "drop_long_gap": True,
        "drop_limit_move_like": True,
        "drop_soft_outlier": True,
        "drop_extreme": True,
        "drop_new_listing_180d": True,
        "max_gap_days": 1.0,
        "max_close_vol_5d": 0.04,
    },
    "stability_focus": {
        "require_trainable": True,
        "drop_long_gap": True,
        "drop_limit_move_like": True,
        "drop_soft_outlier": False,
        "drop_extreme": True,
        "drop_new_listing_180d": False,
        "max_gap_days": 2.0,
        "max_close_vol_5d": 0.03,
    },
    "anti_chase": {
        "require_trainable": True,
        "drop_long_gap": True,
        "drop_limit_move_like": True,
        "drop_soft_outlier": False,
        "drop_extreme": True,
        "drop_new_listing_180d": False,
        "max_gap_days": 1.0,
        "max_close_vol_5d": 0.05,
    },
}


def _to_bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).str.lower().map({"true": True, "false": False}).fillna(False)


def apply_filter_variant(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    out = df.copy()

    if spec.get("require_trainable", False) and "is_trainable" in out.columns:
        out = out.loc[_to_bool_series(out, "is_trainable")].copy()
    if spec.get("drop_long_gap", False):
        out = out.loc[~_to_bool_series(out, "is_long_gap")].copy()
    if spec.get("drop_limit_move_like", False):
        out = out.loc[~_to_bool_series(out, "is_limit_move_like")].copy()
    if spec.get("drop_soft_outlier", False):
        out = out.loc[~_to_bool_series(out, "is_soft_outlier")].copy()
    if spec.get("drop_extreme", False):
        out = out.loc[~_to_bool_series(out, "is_extreme")].copy()
    if spec.get("drop_new_listing_180d", False) and "is_new_listing_180d" in out.columns:
        vals = pd.to_numeric(out["is_new_listing_180d"], errors="coerce").fillna(0)
        out = out.loc[vals < 0.5].copy()

    if spec.get("max_gap_days") is not None and "gap_days" in out.columns:
        out = out.loc[pd.to_numeric(out["gap_days"], errors="coerce") <= float(spec["max_gap_days"])].copy()
    if spec.get("max_close_vol_5d") is not None and "close_vol_5d" in out.columns:
        out = out.loc[pd.to_numeric(out["close_vol_5d"], errors="coerce") <= float(spec["max_close_vol_5d"])].copy()

    return out.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def build_score_specs(weight_override: dict) -> list[tuple[str, float, bool]]:
    specs = []
    for col, weight, ascending in BASE_SCORE_SPECS:
        specs.append((col, weight_override.get(col, weight), ascending))
    return specs


def add_variant_scores(df: pd.DataFrame, score_specs: list[tuple[str, float, bool]]) -> pd.DataFrame:
    out = df.copy()
    out["overnight_score"] = 0.0

    for col, weight, ascending in score_specs:
        if col not in out.columns:
            continue
        series = out[col]
        if series.dtype == bool:
            series = series.astype(float)
        values = pd.to_numeric(series, errors="coerce")
        ranks = values.groupby(out["trade_date"]).rank(pct=True, ascending=ascending)
        out[f"score_component__{col}"] = ranks
        out["overnight_score"] = out["overnight_score"] + float(weight) * ranks.fillna(0.5)

    # lightweight residual penalties even in filtered sets
    penalty = pd.Series(0.0, index=out.index)
    for flag, pen in [("is_long_gap", 0.20), ("is_limit_move_like", 0.20), ("is_soft_outlier", 0.08), ("is_extreme", 0.20)]:
        if flag in out.columns:
            penalty += _to_bool_series(out, flag).astype(float) * pen
    out["overnight_score"] = out["overnight_score"] - penalty
    out["rank_in_day"] = out.groupby("trade_date")["overnight_score"].rank(method="first", ascending=False)
    return out


def _fmt_pct_or_nan(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x:.4%}"


def build_markdown(summary_df: pd.DataFrame, feature_path: Path, top_n: int, exit_mode: str) -> str:
    show = summary_df[[
        "variant", "weight_variant", "filter_variant", "rows_after_filter", "trade_days", "trade_count",
        "total_return", "annualized_return", "max_drawdown", "sharpe", "calmar", "win_rate",
        "avg_trade_return_net", "ending_capital"
    ]].copy()
    for col in ["total_return", "annualized_return", "max_drawdown", "win_rate", "avg_trade_return_net"]:
        show[col] = show[col].map(_fmt_pct_or_nan)
    for col in ["sharpe", "calmar"]:
        show[col] = show[col].map(lambda x: "nan" if pd.isna(x) else f"{x:.4f}")

    best_total = summary_df.sort_values("total_return", ascending=False).head(1)
    best_sharpe = summary_df.sort_values("sharpe", ascending=False).head(1)

    return f"""# Top-{top_n} 因子权重 / 过滤规则批量实验

- Feature input: `{feature_path}`
- Variants: `{len(summary_df)}`
- Top-N: `{top_n}`
- Exit mode: `{exit_mode}`

## Best by total return
{best_total.to_markdown(index=False)}

## Best by Sharpe
{best_sharpe.to_markdown(index=False)}

## Full comparison
{show.to_markdown(index=False)}
""" + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch experiments for Top-5 factor weights and filter rules")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES), help="Unified feature table CSV")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--exit-mode", default=DEFAULT_EXIT_MODE, choices=["open", "0935", "0945", "1000"], help="Exit return column used for evaluation")
    parser.add_argument("--initial-capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--fee-bps", type=float, default=DEFAULT_FEE_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    args = parser.parse_args()

    feature_path = Path(args.features)
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_path}")

    outdir = DEFAULT_OUTDIR
    inputs_dir = outdir / "inputs"
    outdir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    base_df = pd.read_csv(feature_path)
    rows = []

    for weight_name, filter_name in product(WEIGHT_VARIANTS.keys(), FILTER_VARIANTS.keys()):
        variant = f"{weight_name}__{filter_name}"
        filt_df = apply_filter_variant(base_df, FILTER_VARIANTS[filter_name])
        if filt_df.empty:
            rows.append({
                "variant": variant,
                "weight_variant": weight_name,
                "filter_variant": filter_name,
                "rows_after_filter": 0,
                "trade_days": 0,
                "trade_count": 0,
                "ending_capital": float("nan"),
                "total_return": float("nan"),
                "annualized_return": float("nan"),
                "max_drawdown": float("nan"),
                "sharpe": float("nan"),
                "calmar": float("nan"),
                "win_rate": float("nan"),
                "avg_trade_return_net": float("nan"),
                "top_n": args.top_n,
                "status": "empty_after_filter",
            })
            continue

        score_specs = build_score_specs(WEIGHT_VARIANTS[weight_name])
        scored = add_variant_scores(filt_df, score_specs)
        topn_input = build_topn_input(scored, args.top_n)
        input_path = inputs_dir / f"top5_input_{variant}_{feature_path.stem}.csv"
        topn_input.to_csv(input_path, index=False)

        if topn_input.empty:
            rows.append({
                "variant": variant,
                "weight_variant": weight_name,
                "filter_variant": filter_name,
                "rows_after_filter": len(filt_df),
                "trade_days": 0,
                "trade_count": 0,
                "ending_capital": float("nan"),
                "total_return": float("nan"),
                "annualized_return": float("nan"),
                "max_drawdown": float("nan"),
                "sharpe": float("nan"),
                "calmar": float("nan"),
                "win_rate": float("nan"),
                "avg_trade_return_net": float("nan"),
                "top_n": args.top_n,
                "status": "empty_topn",
            })
            continue

        try:
            daily, trades, summary = run_backtest(
                df=topn_input.copy(),
                initial_capital=args.initial_capital,
                fee_bps=args.fee_bps,
                slippage_bps=args.slippage_bps,
                top_n=args.top_n,
                exit_mode=args.exit_mode,
            )
            rows.append({
                "variant": variant,
                "weight_variant": weight_name,
                "filter_variant": filter_name,
                "rows_after_filter": len(filt_df),
                "selected_rows": len(topn_input),
                "status": "ok",
                **summary,
            })
        except Exception as exc:
            rows.append({
                "variant": variant,
                "weight_variant": weight_name,
                "filter_variant": filter_name,
                "rows_after_filter": len(filt_df),
                "selected_rows": len(topn_input),
                "trade_days": 0,
                "trade_count": 0,
                "ending_capital": float("nan"),
                "total_return": float("nan"),
                "annualized_return": float("nan"),
                "max_drawdown": float("nan"),
                "sharpe": float("nan"),
                "calmar": float("nan"),
                "win_rate": float("nan"),
                "avg_trade_return_net": float("nan"),
                "top_n": args.top_n,
                "status": f"error:{type(exc).__name__}",
                "error": str(exc),
            })

    summary_df = pd.DataFrame(rows)
    for col in [
        "total_return", "sharpe", "annualized_return", "max_drawdown", "calmar",
        "win_rate", "avg_trade_return_net", "ending_capital", "trade_days", "trade_count"
    ]:
        if col not in summary_df.columns:
            summary_df[col] = float("nan")
    summary_df = summary_df.sort_values(["total_return", "sharpe"], ascending=[False, False], na_position="last").reset_index(drop=True)

    stem = feature_path.stem.replace("overnight_features_", "") + f"_top{args.top_n}_{args.exit_mode}"
    csv_path = outdir / f"top5_factor_filter_batch_{stem}.csv"
    md_path = outdir / f"top5_factor_filter_batch_{stem}.md"
    summary_df.to_csv(csv_path, index=False)
    md_path.write_text(build_markdown(summary_df, feature_path, args.top_n, args.exit_mode), encoding="utf-8")

    print(f"Wrote experiment CSV: {csv_path} rows={len(summary_df)}")
    print(f"Wrote experiment MD: {md_path}")
    if not summary_df.empty:
        best = summary_df.iloc[0]
        print(
            "Best variant: "
            f"{best.get('variant')} total_return={best.get('total_return')} sharpe={best.get('sharpe')} status={best.get('status')}"
        )


if __name__ == "__main__":
    main()
