#!/usr/bin/env python3
"""Outlier review and clean-label generation for CSI300 overnight labels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/overnight_labels/csi300_overnight_labels_20240101_20260430.csv")
DEFAULT_OUTDIR = Path("data/overnight_labels")


def classify_market(ts_code: str) -> str:
    code = str(ts_code).upper()
    if code.startswith("300") or code.startswith("301"):
        return "ChiNext"
    if code.startswith("688") or code.startswith("689"):
        return "STAR"
    if code.endswith(".BJ") or code.startswith(("8", "4", "9")):
        return "BSE"
    return "Main"


def limit_threshold(market: str, date: str) -> float:
    # Current CSI300 sample starts in 2024, after ChiNext/STAR 20% rules.
    if market in {"ChiNext", "STAR"}:
        return 0.20
    if market == "BSE":
        return 0.30
    return 0.10


def build_clean(df: pd.DataFrame, extreme_abs: float = 0.20, soft_abs: float = 0.08) -> pd.DataFrame:
    out = df.copy()
    out["overnight_return_open"] = pd.to_numeric(out["overnight_return_open"], errors="coerce")
    out["close"] = pd.to_numeric(out["close"], errors="coerce")
    out["next_open"] = pd.to_numeric(out["next_open"], errors="coerce")
    out["gap_days"] = pd.to_numeric(out["gap_days"], errors="coerce")

    out["market_board"] = out["ts_code"].map(classify_market)
    out["limit_threshold"] = [limit_threshold(m, d) for m, d in zip(out["market_board"], out["trade_date"])]
    abs_ret = out["overnight_return_open"].abs()

    # Numeric tolerance captures returns like 0.20000000000000018.
    out["is_extreme"] = abs_ret >= (extreme_abs - 1e-12)
    out["is_soft_outlier"] = abs_ret >= soft_abs
    out["is_limit_move_like"] = abs_ret >= (out["limit_threshold"] - 5e-4)
    out["is_missing_or_invalid"] = (
        out["overnight_return_open"].isna()
        | out["close"].isna()
        | out["next_open"].isna()
        | (out["close"] <= 0)
        | (out["next_open"] <= 0)
    )
    out["is_long_gap"] = out["gap_days"] > 7

    reasons = []
    for row in out.itertuples(index=False):
        r = []
        if row.is_missing_or_invalid:
            r.append("missing_or_invalid_price")
        if row.is_long_gap:
            r.append("long_calendar_gap_gt_7d")
        if row.is_limit_move_like:
            r.append("limit_move_like")
        elif row.is_extreme:
            r.append("extreme_abs_return_ge_20pct")
        elif row.is_soft_outlier:
            r.append("soft_outlier_abs_return_ge_8pct")
        reasons.append(";".join(r))
    out["outlier_reason"] = reasons

    # Trainable excludes hard data issues and limit/extreme-like points, but keeps
    # soft outliers flagged for robustness checks.
    out["is_trainable"] = ~(
        out["is_missing_or_invalid"] | out["is_long_gap"] | out["is_limit_move_like"] | out["is_extreme"]
    )
    return out


def audit(clean: pd.DataFrame, input_path: Path, clean_path: Path, outlier_path: Path) -> dict:
    train = clean[clean["is_trainable"]]
    ret = clean["overnight_return_open"]
    train_ret = train["overnight_return_open"]
    by_reason = clean.loc[clean["outlier_reason"].astype(str) != "", "outlier_reason"].value_counts().to_dict()
    by_board = clean.groupby("market_board").size().to_dict()
    train_by_board = train.groupby("market_board").size().to_dict()
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_path": str(input_path),
        "clean_path": str(clean_path),
        "outlier_review_path": str(outlier_path),
        "n_rows": int(len(clean)),
        "n_symbols": int(clean["ts_code"].nunique()),
        "n_trainable": int(clean["is_trainable"].sum()),
        "n_excluded": int((~clean["is_trainable"]).sum()),
        "n_extreme": int(clean["is_extreme"].sum()),
        "n_soft_outlier": int(clean["is_soft_outlier"].sum()),
        "n_limit_move_like": int(clean["is_limit_move_like"].sum()),
        "n_missing_or_invalid": int(clean["is_missing_or_invalid"].sum()),
        "n_long_gap": int(clean["is_long_gap"].sum()),
        "excluded_by_reason": {str(k): int(v) for k, v in by_reason.items()},
        "rows_by_board": {str(k): int(v) for k, v in by_board.items()},
        "trainable_rows_by_board": {str(k): int(v) for k, v in train_by_board.items()},
        "raw_return_summary": {
            "mean": float(ret.mean()), "std": float(ret.std()), "min": float(ret.min()),
            "p01": float(ret.quantile(0.01)), "p05": float(ret.quantile(0.05)),
            "median": float(ret.median()), "p95": float(ret.quantile(0.95)),
            "p99": float(ret.quantile(0.99)), "max": float(ret.max()),
        },
        "trainable_return_summary": {
            "mean": float(train_ret.mean()), "std": float(train_ret.std()), "min": float(train_ret.min()),
            "p01": float(train_ret.quantile(0.01)), "p05": float(train_ret.quantile(0.05)),
            "median": float(train_ret.median()), "p95": float(train_ret.quantile(0.95)),
            "p99": float(train_ret.quantile(0.99)), "max": float(train_ret.max()),
        },
    }


def write_md(a: dict, hard_outliers: pd.DataFrame, soft_outliers: pd.DataFrame, path: Path) -> None:
    lines = [
        "# CSI300 Overnight Clean Label Outlier Review",
        "",
        f"- Created at: `{a['created_at']}`",
        f"- Input: `{a['input_path']}`",
        f"- Clean labels: `{a['clean_path']}`",
        f"- Outlier review CSV: `{a['outlier_review_path']}`",
        "",
        "## Cleaning Rule",
        "",
        "- `is_extreme`: `abs(overnight_return_open) >= 20%`",
        "- `is_limit_move_like`: return near board-specific daily limit (`10%` main board, `20%` ChiNext/STAR, `30%` BSE)",
        "- `is_soft_outlier`: `abs(return) >= 8%`, retained for training but flagged",
        "- `is_trainable`: excludes missing/invalid prices, long gaps > 7 calendar days, limit-move-like rows, and hard extremes",
        "",
        "## Summary",
        "",
        f"- Total rows: `{a['n_rows']}`",
        f"- Symbols: `{a['n_symbols']}`",
        f"- Trainable rows: `{a['n_trainable']}`",
        f"- Excluded rows: `{a['n_excluded']}`",
        f"- Hard extreme rows: `{a['n_extreme']}`",
        f"- Limit-move-like rows: `{a['n_limit_move_like']}`",
        f"- Soft outlier rows: `{a['n_soft_outlier']}`",
        f"- Missing/invalid rows: `{a['n_missing_or_invalid']}`",
        f"- Long gap rows: `{a['n_long_gap']}`",
        "",
        "## Excluded by Reason",
        "",
    ]
    if a["excluded_by_reason"]:
        for k, v in a["excluded_by_reason"].items():
            lines.append(f"- `{k}`: `{v}`")
    else:
        lines.append("- None")

    lines += ["", "## Raw vs Trainable Return Summary", "", "| Metric | Raw | Trainable |", "|---|---:|---:|"]
    for k in ["mean", "std", "min", "p01", "p05", "median", "p95", "p99", "max"]:
        lines.append(f"| `{k}` | `{a['raw_return_summary'][k]}` | `{a['trainable_return_summary'][k]}` |")

    lines += ["", "## Hard Outliers / Limit-like Rows", ""]
    if hard_outliers.empty:
        lines.append("- None")
    else:
        for r in hard_outliers.head(80).itertuples(index=False):
            lines.append(
                f"- `{r.ts_code}` `{r.trade_date}` -> `{r.next_trade_date}` "
                f"return=`{r.overnight_return_open}` board=`{r.market_board}` reason=`{r.outlier_reason}`"
            )

    lines += ["", "## Soft Outlier Sample", ""]
    if soft_outliers.empty:
        lines.append("- None")
    else:
        for r in soft_outliers.head(40).itertuples(index=False):
            lines.append(
                f"- `{r.ts_code}` `{r.trade_date}` return=`{r.overnight_return_open}` "
                f"board=`{r.market_board}` reason=`{r.outlier_reason}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--extreme-abs", type=float, default=0.20)
    parser.add_argument("--soft-abs", type=float, default=0.08)
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tag = input_path.stem.replace("csi300_overnight_labels_", "")

    df = pd.read_csv(input_path)
    clean = build_clean(df, extreme_abs=args.extreme_abs, soft_abs=args.soft_abs)

    clean_path = outdir / f"csi300_overnight_labels_clean_{tag}.csv"
    outlier_path = outdir / f"csi300_overnight_outlier_review_{tag}.csv"
    audit_json = outdir / f"csi300_overnight_clean_audit_{tag}.json"
    audit_md = outdir / f"csi300_overnight_clean_audit_{tag}.md"

    hard = clean[(clean["is_extreme"]) | (clean["is_limit_move_like"]) | (clean["is_missing_or_invalid"]) | (clean["is_long_gap"])].copy()
    soft = clean[(clean["is_soft_outlier"]) & (~clean.index.isin(hard.index))].copy()
    review = pd.concat([hard, soft], ignore_index=True).sort_values(["is_trainable", "ts_code", "trade_date"])

    clean.to_csv(clean_path, index=False)
    review.to_csv(outlier_path, index=False)
    a = audit(clean, input_path, clean_path, outlier_path)
    audit_json.write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
    write_md(a, hard.sort_values(["ts_code", "trade_date"]), soft.sort_values(["ts_code", "trade_date"]), audit_md)

    print("DONE")
    print(f"clean: {clean_path}")
    print(f"outlier_review: {outlier_path}")
    print(f"audit_md: {audit_md}")
    print(json.dumps({k: a[k] for k in ["n_rows", "n_trainable", "n_excluded", "n_extreme", "n_limit_move_like", "n_soft_outlier", "n_missing_or_invalid", "n_long_gap"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
