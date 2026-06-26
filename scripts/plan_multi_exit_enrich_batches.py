#!/usr/bin/env python3
"""Plan concrete enrich batches from priority-row CSV.

This script turns the output of audit_multi_exit_coverage.py into small,
quota-aware batches for enrich_overnight_labels_multi_exit.py.

Outputs:
  data/overnight_labels/coverage_audit/enrich_batches_<stem>.csv
  data/overnight_labels/coverage_audit/enrich_batches_<stem>.md
  data/overnight_labels/coverage_audit/enrich_batches_<stem>.sh
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

DEFAULT_PRIORITY = Path("data/overnight_labels/coverage_audit/multi_exit_priority_rows_20240101_20260430__recent5.csv")
DEFAULT_CLEAN = Path("data/overnight_labels/csi300_overnight_labels_clean_20240101_20260430.csv")
DEFAULT_MULTI = Path("data/overnight_labels/csi300_overnight_labels_multi_exit_20240101_20260430.csv")
DEFAULT_AUDIT = Path("data/overnight_labels/csi300_overnight_labels_multi_exit_audit_20240101_20260430.md")
DEFAULT_OUTDIR = Path("data/overnight_labels/coverage_audit")


def _split_csv(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def _shell_quote_single(text: str) -> str:
    return "'" + str(text).replace("'", "'\"'\"'") + "'"


def _parse_iso_like(value) -> datetime | None:
    text = str(value).strip()
    if not text or text in {"<NA>", "nan", "None"}:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def load_priority(path: Path, limit_rows: int, buckets: list[str], skip_recent_error_days: int, as_of_date: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Priority CSV not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    if buckets:
        df = df.loc[df["priority_bucket"].astype(str).isin(buckets)].copy()
    if skip_recent_error_days > 0:
        cutoff = datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=skip_recent_error_days)
        attempted_ts = df.get("minute_attempted_at", pd.Series(pd.NA, index=df.index)).map(_parse_iso_like)
        has_recent_error = (
            df.get("has_minute_error", pd.Series(False, index=df.index)).fillna(False).astype(bool)
            & attempted_ts.notna()
            & (attempted_ts >= cutoff)
        )
        df = df.loc[~has_recent_error].copy()
    if limit_rows and limit_rows > 0:
        df = df.head(limit_rows).copy()
    if df.empty:
        raise ValueError("No priority rows left after filters")
    return df.reset_index(drop=True)


def build_batches(df: pd.DataFrame, grouping: str, batch_size: int) -> list[dict]:
    rows = []
    batch_id = 1

    if grouping == "date":
        for trade_date, g in df.groupby("trade_date", sort=False):
            symbols = g["ts_code"].astype(str).tolist()
            for i in range(0, len(symbols), batch_size):
                chunk = symbols[i:i + batch_size]
                rows.append({
                    "batch_id": batch_id,
                    "grouping": grouping,
                    "trade_date": str(trade_date),
                    "symbol_count": len(chunk),
                    "symbols": ",".join(chunk),
                    "priority_bucket_mix": ",".join(sorted(g.iloc[i:i + batch_size]["priority_bucket"].astype(str).unique().tolist())),
                })
                batch_id += 1
    elif grouping == "symbol":
        symbols = df["ts_code"].astype(str).tolist()
        dates = df["trade_date"].astype(str).tolist()
        for i in range(0, len(symbols), batch_size):
            chunk = df.iloc[i:i + batch_size].copy()
            rows.append({
                "batch_id": batch_id,
                "grouping": grouping,
                "trade_date": ",".join(sorted(chunk["trade_date"].astype(str).unique().tolist())),
                "symbol_count": len(chunk),
                "symbols": ",".join(chunk["ts_code"].astype(str).tolist()),
                "priority_bucket_mix": ",".join(sorted(chunk["priority_bucket"].astype(str).unique().tolist())),
            })
            batch_id += 1
    else:
        raise ValueError(f"Unsupported grouping={grouping!r}")

    return rows


def build_command(row: dict, clean_path: Path, multi_path: Path, audit_path: Path, retries: int, rate_limit_sleep: float, cooldown_days: int) -> str:
    trade_date = str(row["trade_date"]).split(",")[0]
    symbols = str(row["symbols"])
    return (
        "python3 scripts/enrich_overnight_labels_multi_exit.py "
        f"--input {_shell_quote_single(str(clean_path))} "
        f"--output {_shell_quote_single(str(multi_path))} "
        f"--audit {_shell_quote_single(str(audit_path))} "
        f"--start-date {_shell_quote_single(trade_date)} "
        f"--end-date {_shell_quote_single(trade_date)} "
        f"--symbols {_shell_quote_single(symbols)} "
        "--batch-limit 0 "
        "--prefer-no-error-first "
        "--prioritize-recent "
        f"--cooldown-days {int(cooldown_days)} "
        f"--retries {int(retries)} "
        f"--rate-limit-sleep {float(rate_limit_sleep)}"
    )


def build_markdown(priority_path: Path, batches_df: pd.DataFrame, *, skip_recent_error_days: int, as_of_date: str) -> str:
    show = batches_df[["batch_id", "grouping", "trade_date", "symbol_count", "priority_bucket_mix", "symbols"]].copy()
    return f"""# Planned Multi-Exit Enrich Batches

- Priority input: `{priority_path}`
- Batch count: `{len(batches_df)}`
- Skip recent error days: `{skip_recent_error_days}`
- As-of date: `{as_of_date}`

## Batch summary
{show.to_markdown(index=False)}
""" + "\n"


def build_shell_script(batches_df: pd.DataFrame) -> str:
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    for _, row in batches_df.iterrows():
        lines.append(f"# batch {int(row['batch_id'])}: trade_date={row['trade_date']} symbols={int(row['symbol_count'])}")
        lines.append(str(row["command"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan concrete enrich batches from priority-row CSV")
    parser.add_argument("--priority", default=str(DEFAULT_PRIORITY), help="Priority rows CSV from audit_multi_exit_coverage.py")
    parser.add_argument("--clean", default=str(DEFAULT_CLEAN), help="Authoritative clean label CSV")
    parser.add_argument("--multi", default=str(DEFAULT_MULTI), help="Target multi-exit label CSV")
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT), help="Target enrich audit markdown")
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help="Output directory")
    parser.add_argument("--grouping", choices=["date", "symbol"], default="date", help="How to form batches")
    parser.add_argument("--batch-size", type=int, default=2, help="Symbols per batch when grouping by date, or rows per batch when grouping by symbol")
    parser.add_argument("--limit-rows", type=int, default=20, help="Only plan from the first N priority rows")
    parser.add_argument("--buckets", default="fresh_missing", help="Comma-separated priority buckets to include")
    parser.add_argument("--retries", type=int, default=0, help="Retries for stk_mins after rate limits")
    parser.add_argument("--rate-limit-sleep", type=float, default=31.0, help="Backoff seconds when rate-limited")
    parser.add_argument("--cooldown-days", type=int, default=1, help="Cooldown-days argument to pass through")
    parser.add_argument("--skip-recent-error-days", type=int, default=1, help="Exclude rows whose minute_error was attempted within the last N days")
    parser.add_argument("--as-of-date", default=datetime.now().strftime("%Y-%m-%d"), help="Logical YYYY-MM-DD used by --skip-recent-error-days")
    args = parser.parse_args()

    priority_path = Path(args.priority)
    clean_path = Path(args.clean)
    multi_path = Path(args.multi)
    audit_path = Path(args.audit)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    buckets = _split_csv(args.buckets)
    priority_df = load_priority(
        priority_path,
        args.limit_rows,
        buckets,
        skip_recent_error_days=args.skip_recent_error_days,
        as_of_date=args.as_of_date,
    )
    batches = build_batches(priority_df, grouping=args.grouping, batch_size=args.batch_size)
    batches_df = pd.DataFrame(batches)
    batches_df["command"] = batches_df.apply(
        lambda r: build_command(
            r.to_dict(),
            clean_path=clean_path,
            multi_path=multi_path,
            audit_path=audit_path,
            retries=args.retries,
            rate_limit_sleep=args.rate_limit_sleep,
            cooldown_days=args.cooldown_days,
        ),
        axis=1,
    )

    stem = priority_path.stem.replace("multi_exit_priority_rows_", "")
    stem = f"{stem}__skiperr{args.skip_recent_error_days}_{args.as_of_date}__{args.grouping}_b{args.batch_size}_n{args.limit_rows}"
    csv_path = outdir / f"enrich_batches_{stem}.csv"
    md_path = outdir / f"enrich_batches_{stem}.md"
    sh_path = outdir / f"enrich_batches_{stem}.sh"

    batches_df.to_csv(csv_path, index=False)
    md_path.write_text(
        build_markdown(
            priority_path,
            batches_df,
            skip_recent_error_days=args.skip_recent_error_days,
            as_of_date=args.as_of_date,
        ),
        encoding="utf-8",
    )
    sh_path.write_text(build_shell_script(batches_df), encoding="utf-8")

    print(f"Wrote batch CSV: {csv_path} rows={len(batches_df)}")
    print(f"Wrote batch MD: {md_path}")
    print(f"Wrote batch shell: {sh_path}")
    if not batches_df.empty:
        first = batches_df.iloc[0]
        print(f"First batch: id={first['batch_id']} trade_date={first['trade_date']} symbols={first['symbols']}")


if __name__ == "__main__":
    main()
