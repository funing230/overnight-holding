#!/usr/bin/env python3
"""Audit multi-exit label coverage and prioritize minute-enrichment targets.

This script is designed for low-quota Tushare workflows where the multi-exit
label file may be partially enriched or even accidentally subset-written.
It uses the clean overnight label table as the authoritative universe and
optionally overlays any existing multi-exit output on matching (ts_code,
trade_date) keys.

Outputs:
  data/overnight_labels/coverage_audit/multi_exit_coverage_summary_<stem>.md
  data/overnight_labels/coverage_audit/multi_exit_coverage_by_date_<stem>.csv
  data/overnight_labels/coverage_audit/multi_exit_coverage_by_symbol_<stem>.csv
  data/overnight_labels/coverage_audit/multi_exit_priority_rows_<stem>.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CLEAN = Path("data/overnight_labels/csi300_overnight_labels_clean_20240101_20260430.csv")
DEFAULT_MULTI = Path("data/overnight_labels/csi300_overnight_labels_multi_exit_20240101_20260430.csv")
DEFAULT_OUTDIR = Path("data/overnight_labels/coverage_audit")
DEFAULT_EXIT_MODES = ["0935", "0945", "1000"]

EXIT_RETURN_COLS = {
    "0935": "overnight_return_0935",
    "0945": "overnight_return_0945",
    "1000": "overnight_return_1000",
}
EXIT_PRICE_COLS = {
    "0935": "next_close_0935",
    "0945": "next_close_0945",
    "1000": "next_close_1000",
}


def _fmt_pct(x: float) -> str:
    return "nan" if pd.isna(x) else f"{x:.4%}"


def _is_nonempty_error(v) -> bool:
    if pd.isna(v):
        return False
    return str(v).strip() not in {"", "<NA>", "nan", "None"}


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def load_base(clean_path: Path, start_date: str, end_date: str, symbols: list[str]) -> pd.DataFrame:
    if not clean_path.exists():
        raise FileNotFoundError(f"Clean labels not found: {clean_path}")
    df = pd.read_csv(clean_path)
    if start_date:
        df = df.loc[df["trade_date"].astype(str) >= start_date].copy()
    if end_date:
        df = df.loc[df["trade_date"].astype(str) <= end_date].copy()
    if symbols:
        wanted = {s.upper() for s in symbols}
        df = df.loc[df["ts_code"].astype(str).str.upper().isin(wanted)].copy()
    if df.empty:
        raise ValueError("No rows left in clean labels after filters")
    return df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)


def overlay_multi_exit(base: pd.DataFrame, multi_path: Path, exit_modes: list[str]) -> tuple[pd.DataFrame, dict]:
    out = base.copy()
    diagnostics = {
        "multi_exists": multi_path.exists(),
        "multi_rows": 0,
        "overlay_matches": 0,
        "multi_key_coverage": 0.0,
    }

    for col in ["minute_source", "minute_error", "minute_attempted_at"] + [EXIT_RETURN_COLS[m] for m in exit_modes] + [EXIT_PRICE_COLS[m] for m in exit_modes]:
        if col not in out.columns:
            out[col] = pd.NA

    if not multi_path.exists():
        out["has_multi_row"] = False
        return out, diagnostics

    multi = pd.read_csv(multi_path)
    diagnostics["multi_rows"] = int(len(multi))
    if multi.empty:
        out["has_multi_row"] = False
        return out, diagnostics

    keep = [c for c in ["ts_code", "trade_date", "minute_source", "minute_error", "minute_attempted_at"] + [EXIT_RETURN_COLS[m] for m in exit_modes] + [EXIT_PRICE_COLS[m] for m in exit_modes] if c in multi.columns]
    multi = multi[keep].copy().drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    out = out.merge(multi, on=["ts_code", "trade_date"], how="left", suffixes=("", "__multi"))

    matched = out["trade_date"].notna()
    diagnostics["overlay_matches"] = int(out[["ts_code", "trade_date"]].merge(multi[["ts_code", "trade_date"]], on=["ts_code", "trade_date"], how="inner").shape[0])
    diagnostics["multi_key_coverage"] = float(diagnostics["overlay_matches"] / len(out)) if len(out) else float("nan")

    for col in ["minute_source", "minute_error", "minute_attempted_at"] + [EXIT_RETURN_COLS[m] for m in exit_modes] + [EXIT_PRICE_COLS[m] for m in exit_modes]:
        mcol = f"{col}__multi"
        if mcol in out.columns:
            if col in base.columns:
                out[col] = out[mcol].combine_first(out[col])
            else:
                out[col] = out[mcol]
            out = out.drop(columns=[mcol])

    multi_keys = set(zip(multi["ts_code"].astype(str), multi["trade_date"].astype(str)))
    out["has_multi_row"] = [((str(ts), str(dt)) in multi_keys) for ts, dt in zip(out["ts_code"], out["trade_date"])]
    return out, diagnostics


def add_status_flags(df: pd.DataFrame, exit_modes: list[str]) -> pd.DataFrame:
    out = df.copy()
    for mode in exit_modes:
        ret_col = EXIT_RETURN_COLS[mode]
        out[f"has_exit_{mode}"] = pd.to_numeric(out.get(ret_col), errors="coerce").notna()
    out["has_any_exit"] = out[[f"has_exit_{m}" for m in exit_modes]].any(axis=1)
    out["has_all_exits"] = out[[f"has_exit_{m}" for m in exit_modes]].all(axis=1)
    out["has_minute_error"] = out.get("minute_error", pd.Series(pd.NA, index=out.index)).map(_is_nonempty_error)
    out["attempted_before"] = out.get("minute_attempted_at", pd.Series(pd.NA, index=out.index)).notna()
    return out


def summarize_by_date(df: pd.DataFrame, exit_modes: list[str], focus_recent_days: int = 0) -> pd.DataFrame:
    rows = []
    all_dates_sorted = sorted(pd.Series(df["trade_date"].astype(str).dropna().unique()).tolist())
    recent_focus_dates = set(all_dates_sorted[-focus_recent_days:]) if focus_recent_days and focus_recent_days > 0 else set()
    for trade_date, g in df.groupby("trade_date", sort=True):
        row = {
            "trade_date": trade_date,
            "rows": int(len(g)),
            "has_multi_row_rows": int(g["has_multi_row"].sum()),
            "has_any_exit_rows": int(g["has_any_exit"].sum()),
            "has_all_exits_rows": int(g["has_all_exits"].sum()),
            "minute_error_rows": int(g["has_minute_error"].sum()),
            "attempted_before_rows": int(g["attempted_before"].sum()),
            "is_recent_focus": str(trade_date) in recent_focus_dates,
        }
        row["coverage_any_exit"] = row["has_any_exit_rows"] / row["rows"] if row["rows"] else float("nan")
        row["coverage_all_exits"] = row["has_all_exits_rows"] / row["rows"] if row["rows"] else float("nan")
        for mode in exit_modes:
            count = int(g[f"has_exit_{mode}"].sum())
            row[f"rows_{mode}"] = count
            row[f"coverage_{mode}"] = count / row["rows"] if row["rows"] else float("nan")
        recent_bonus = 10000.0 if row["is_recent_focus"] else 0.0
        row["priority_score"] = (
            recent_bonus
            + (1.0 - row["coverage_all_exits"]) * 1000.0
            + min(row["rows"], 500)
            - row["minute_error_rows"] * 0.1
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["priority_score", "trade_date"], ascending=[False, False]).reset_index(drop=True)


def summarize_by_symbol(df: pd.DataFrame, exit_modes: list[str]) -> pd.DataFrame:
    rows = []
    for ts_code, g in df.groupby("ts_code", sort=True):
        row = {
            "ts_code": ts_code,
            "rows": int(len(g)),
            "has_multi_row_rows": int(g["has_multi_row"].sum()),
            "has_any_exit_rows": int(g["has_any_exit"].sum()),
            "has_all_exits_rows": int(g["has_all_exits"].sum()),
            "minute_error_rows": int(g["has_minute_error"].sum()),
            "attempted_before_rows": int(g["attempted_before"].sum()),
            "date_min": str(g["trade_date"].min()),
            "date_max": str(g["trade_date"].max()),
        }
        row["coverage_any_exit"] = row["has_any_exit_rows"] / row["rows"] if row["rows"] else float("nan")
        row["coverage_all_exits"] = row["has_all_exits_rows"] / row["rows"] if row["rows"] else float("nan")
        for mode in exit_modes:
            count = int(g[f"has_exit_{mode}"].sum())
            row[f"rows_{mode}"] = count
            row[f"coverage_{mode}"] = count / row["rows"] if row["rows"] else float("nan")
        row["priority_score"] = (
            (1.0 - row["coverage_all_exits"]) * 100.0
            + min(row["rows"], 50)
            - row["minute_error_rows"] * 0.1
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["priority_score", "rows", "ts_code"], ascending=[False, False, True]).reset_index(drop=True)


def build_priority_rows(df: pd.DataFrame, exit_modes: list[str], prioritize_recent: bool, max_rows: int, focus_recent_days: int = 0) -> pd.DataFrame:
    need_cols = [f"has_exit_{m}" for m in exit_modes]
    out = df.loc[~df[need_cols].all(axis=1)].copy()
    if focus_recent_days and focus_recent_days > 0:
        all_dates_sorted = sorted(pd.Series(out["trade_date"].astype(str).dropna().unique()).tolist())
        recent_focus_dates = set(all_dates_sorted[-focus_recent_days:])
        out["is_recent_focus"] = out["trade_date"].astype(str).isin(recent_focus_dates)
        out = out.loc[out["is_recent_focus"]].copy()
    else:
        out["is_recent_focus"] = False
    out["missing_exit_count"] = out[need_cols].apply(lambda r: int((~r).sum()), axis=1)
    out["missing_exit_modes"] = out.apply(lambda r: ",".join([m for m in exit_modes if not bool(r[f"has_exit_{m}"])]), axis=1)
    out["priority_bucket"] = out.apply(
        lambda r: "fresh_missing" if (not r["attempted_before"] and not r["has_minute_error"]) else ("retry_error" if r["has_minute_error"] else "partial_missing"),
        axis=1,
    )
    sort_cols = ["priority_bucket", "missing_exit_count", "trade_date", "ts_code"]
    ascending = [True, False, not prioritize_recent, True]
    out = out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
    keep = [
        "ts_code", "trade_date", "next_trade_date", "close", "next_open", "overnight_return_open",
        "priority_bucket", "is_recent_focus", "missing_exit_count", "missing_exit_modes",
        "has_multi_row", "attempted_before", "has_minute_error", "minute_attempted_at", "minute_error",
    ] + [EXIT_RETURN_COLS[m] for m in exit_modes if EXIT_RETURN_COLS[m] in out.columns]
    return out[keep].head(max_rows)


def build_markdown(base_path: Path, multi_path: Path, exit_modes: list[str], df: pd.DataFrame, by_date: pd.DataFrame, by_symbol: pd.DataFrame, priority_rows: pd.DataFrame, diagnostics: dict, focus_recent_days: int) -> str:
    total = len(df)
    any_exit = int(df["has_any_exit"].sum())
    all_exits = int(df["has_all_exits"].sum())
    minute_errors = int(df["has_minute_error"].sum())
    attempted = int(df["attempted_before"].sum())

    exit_lines = []
    for mode in exit_modes:
        cnt = int(df[f"has_exit_{mode}"].sum())
        exit_lines.append(f"- `{mode}` coverage: `{cnt}/{total}` = `{_fmt_pct(cnt / total if total else float('nan'))}`")

    top_dates = by_date.head(10).copy()
    top_symbols = by_symbol.head(10).copy()
    for col in [c for c in top_dates.columns if c.startswith("coverage_")]:
        top_dates[col] = top_dates[col].map(_fmt_pct)
    for col in [c for c in top_symbols.columns if c.startswith("coverage_")]:
        top_symbols[col] = top_symbols[col].map(_fmt_pct)

    return f"""# Multi-Exit Coverage Audit

- Clean base: `{base_path}`
- Multi-exit overlay: `{multi_path}`
- Base rows in scope: `{total}`
- Multi-exit file exists: `{diagnostics['multi_exists']}`
- Multi-exit file rows: `{diagnostics['multi_rows']}`
- Overlay key matches: `{diagnostics['overlay_matches']}`
- Overlay key coverage: `{_fmt_pct(diagnostics['multi_key_coverage'])}`
- Focus recent days: `{focus_recent_days}`
- Rows with any minute exit: `{any_exit}` / `{total}` = `{_fmt_pct(any_exit / total if total else float('nan'))}`
- Rows with all minute exits: `{all_exits}` / `{total}` = `{_fmt_pct(all_exits / total if total else float('nan'))}`
- Rows attempted before: `{attempted}`
- Rows with minute_error: `{minute_errors}`

## Exit coverage snapshot
{chr(10).join(exit_lines)}

## Top priority trade dates
{top_dates[['trade_date','is_recent_focus','rows','has_multi_row_rows','has_all_exits_rows','minute_error_rows','coverage_all_exits','priority_score']].to_markdown(index=False) if not top_dates.empty else 'No rows'}

## Top priority symbols
{top_symbols[['ts_code','rows','has_multi_row_rows','has_all_exits_rows','minute_error_rows','coverage_all_exits','priority_score']].to_markdown(index=False) if not top_symbols.empty else 'No rows'}

## First priority rows to enrich
{priority_rows.head(20).to_markdown(index=False) if not priority_rows.empty else 'No missing rows'}
""" + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit multi-exit label coverage and prioritize minute-enrichment targets")
    parser.add_argument("--clean", default=str(DEFAULT_CLEAN), help="Authoritative clean label CSV")
    parser.add_argument("--multi", default=str(DEFAULT_MULTI), help="Current multi-exit label CSV")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory")
    parser.add_argument("--start-date", default="", help="Optional trade_date >= YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="Optional trade_date <= YYYY-MM-DD")
    parser.add_argument("--symbols", default="", help="Optional comma-separated ts_code filter")
    parser.add_argument("--exit-modes", default=",".join(DEFAULT_EXIT_MODES), help="Comma-separated exit modes, e.g. 0935,0945,1000")
    parser.add_argument("--focus-recent-days", type=int, default=0, help="When >0, prioritize and row-filter to the most recent N trade dates in scope")
    parser.add_argument("--prioritize-recent", action="store_true", help="Sort row-level priorities by newer trade_date first within priority buckets")
    parser.add_argument("--max-priority-rows", type=int, default=200, help="Max priority rows to write")
    args = parser.parse_args()

    clean_path = Path(args.clean)
    multi_path = Path(args.multi)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    exit_modes = _split_csv(args.exit_modes)
    unsupported = [m for m in exit_modes if m not in EXIT_RETURN_COLS]
    if unsupported:
        raise ValueError(f"Unsupported exit modes: {unsupported}")

    symbols = _split_csv(args.symbols)
    base = load_base(clean_path, args.start_date, args.end_date, symbols)
    merged, diagnostics = overlay_multi_exit(base, multi_path, exit_modes)
    merged = add_status_flags(merged, exit_modes)

    by_date = summarize_by_date(merged, exit_modes, focus_recent_days=args.focus_recent_days)
    by_symbol = summarize_by_symbol(merged, exit_modes)
    priority_rows = build_priority_rows(
        merged,
        exit_modes,
        prioritize_recent=args.prioritize_recent,
        max_rows=args.max_priority_rows,
        focus_recent_days=args.focus_recent_days,
    )

    stem_parts = []
    stem_parts.append(clean_path.stem.replace("csi300_overnight_labels_clean_", "") or clean_path.stem)
    if args.start_date or args.end_date:
        stem_parts.append((args.start_date or "start") + "_" + (args.end_date or "end"))
    if args.focus_recent_days and args.focus_recent_days > 0:
        stem_parts.append(f"recent{args.focus_recent_days}")
    if symbols:
        stem_parts.append(f"symbols{len(symbols)}")
    stem = "__".join(stem_parts)

    date_path = outdir / f"multi_exit_coverage_by_date_{stem}.csv"
    symbol_path = outdir / f"multi_exit_coverage_by_symbol_{stem}.csv"
    priority_path = outdir / f"multi_exit_priority_rows_{stem}.csv"
    summary_path = outdir / f"multi_exit_coverage_summary_{stem}.md"

    by_date.to_csv(date_path, index=False)
    by_symbol.to_csv(symbol_path, index=False)
    priority_rows.to_csv(priority_path, index=False)
    summary_path.write_text(build_markdown(clean_path, multi_path, exit_modes, merged, by_date, by_symbol, priority_rows, diagnostics, args.focus_recent_days), encoding="utf-8")

    print(f"Wrote date coverage: {date_path} rows={len(by_date)}")
    print(f"Wrote symbol coverage: {symbol_path} rows={len(by_symbol)}")
    print(f"Wrote priority rows: {priority_path} rows={len(priority_rows)}")
    print(f"Wrote summary: {summary_path}")
    print(f"Any-exit coverage={merged['has_any_exit'].mean():.6f} All-exit coverage={merged['has_all_exits'].mean():.6f}")


if __name__ == "__main__":
    main()
