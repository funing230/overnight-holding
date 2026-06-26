#!/usr/bin/env python3
"""Incrementally enrich overnight labels with cached T+1 minute-exit prices.

This script is designed for low-quota Tushare ``stk_mins`` environments.
It keeps the existing daily overnight label pipeline intact and adds a resumable
post-processing step that:

- reads an existing overnight label CSV
- fills `09:35 / 09:45 / 10:00` exit prices and returns
- caches minute bars under `data/tushare_minute_cache/`
- skips rows that already have all requested exits
- can stop after a small batch to stay within token limits
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from dataflows.ashare_overnight_labels import (
    DEFAULT_EXIT_TIMES,
    DEFAULT_MINUTE_CACHE_DIR,
    enrich_overnight_row_with_intraday_exits,
)


DEFAULT_INPUT = Path("data/overnight_labels/csi300_overnight_labels_clean_20240101_20260430.csv")
DEFAULT_OUTPUT = Path("data/overnight_labels/csi300_overnight_labels_multi_exit_20240101_20260430.csv")
DEFAULT_AUDIT = Path("data/overnight_labels/csi300_overnight_labels_multi_exit_audit_20240101_20260430.md")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _suffix(exit_time: str) -> str:
    return exit_time.replace(":", "")[:4]


def _is_complete(row: pd.Series, exit_times: list[str]) -> bool:
    for exit_time in exit_times:
        if pd.isna(row.get(f"next_close_{_suffix(exit_time)}")):
            return False
    return True


def _has_minute_error(row: pd.Series) -> bool:
    value = row.get("minute_error")
    if pd.isna(value):
        return False
    return str(value).strip() not in {"", "<NA>", "nan", "None"}


def _is_nonempty_error_value(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip() not in {"", "<NA>", "nan", "None"}


def _normalize_run_date(value: str) -> str:
    text = str(value).strip()
    if not text:
        return datetime.now().strftime("%Y-%m-%d")
    return text


def _parse_iso_like(value: str) -> datetime | None:
    text = str(value).strip()
    if not text or text in {"<NA>", "nan", "None"}:
        return None
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def _should_skip_due_to_cooldown(row: pd.Series, run_date: str, cooldown_days: int) -> bool:
    if cooldown_days <= 0:
        return False
    if not _has_minute_error(row):
        return False
    last_attempt = _parse_iso_like(row.get("minute_attempted_at"))
    if last_attempt is None:
        return False
    run_dt = datetime.strptime(run_date, "%Y-%m-%d")
    return last_attempt >= (run_dt - timedelta(days=cooldown_days))


def write_audit(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    cache_dir: Path,
    exit_times: list[str],
    total_rows: int,
    already_complete_rows: int,
    cooldown_skipped_rows: int,
    attempted_rows: int,
    newly_completed_rows: int,
    minute_error_rows: int,
    cache_files: int,
    batch_limit: int,
    prioritize_recent: bool,
    prefer_no_error_first: bool,
    cooldown_days: int,
    run_date: str,
) -> None:
    text = f"""# Overnight Multi-Exit Enrichment Audit

- Created at: `{datetime.now().isoformat(timespec='seconds')}`
- Input CSV: `{input_path}`
- Output CSV: `{output_path}`
- Minute cache dir: `{cache_dir}`
- Exit times: `{', '.join(exit_times)}`
- Total rows seen: `{total_rows}`
- Already complete rows skipped: `{already_complete_rows}`
- Cooldown-skipped minute_error rows: `{cooldown_skipped_rows}`
- Rows attempted this run: `{attempted_rows}`
- Rows newly completed this run: `{newly_completed_rows}`
- Rows with minute_error after this run: `{minute_error_rows}`
- Cache files present: `{cache_files}`
- Batch limit used: `{batch_limit}`
        prioritize_recent: `{prioritize_recent}`
- Prefer no-error-first: `{prefer_no_error_first}`
- Cooldown days: `{cooldown_days}`
- Run date: `{run_date}`

## Notes
- This script is resumable: rerun it to continue enriching unfinished rows.
- Cached `stk_mins` results are reused from disk to avoid spending quota twice.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally enrich overnight labels with cached multi-exit minute bars.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Existing overnight label CSV")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Enriched output CSV")
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT), help="Audit markdown output")
    parser.add_argument("--exit-times", default=",".join(DEFAULT_EXIT_TIMES), help="Comma-separated exit times, e.g. 09:35,09:45,10:00")
    parser.add_argument("--cache-dir", default=str(DEFAULT_MINUTE_CACHE_DIR), help="Minute-bar cache directory")
    parser.add_argument("--batch-limit", type=int, default=2, help="Max unfinished rows to attempt this run; 0 means all")
    parser.add_argument("--start-date", default="", help="Optional filter on trade_date >= YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="Optional filter on trade_date <= YYYY-MM-DD")
    parser.add_argument("--symbols", default="", help="Optional comma-separated ts_code filter")
    parser.add_argument("--retries", type=int, default=0, help="Retries for stk_mins after rate limits")
    parser.add_argument("--rate-limit-sleep", type=float, default=31.0, help="Backoff seconds when rate-limited")
    parser.add_argument("--prioritize-recent", action="store_true", help="Attempt newest trade_date rows first")
    parser.add_argument("--prefer-no-error-first", action="store_true", help="Prioritize rows with no prior minute_error before retrying failed rows")
    parser.add_argument("--cooldown-days", type=int, default=1, help="Skip rows with existing minute_error attempted within this many days")
    parser.add_argument("--run-date", default="", help="Logical run date YYYY-MM-DD for cooldown decisions; defaults to today")
    parser.add_argument("--minute-provider", choices=["auto", "tushare", "akshare"], default="auto", help="Minute-bar provider; auto tries Tushare then AkShare fallback")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    audit_path = Path(args.audit)
    cache_dir = Path(args.cache_dir)
    exit_times = _split_csv(args.exit_times)
    run_date = _normalize_run_date(args.run_date)

    if not input_path.exists():
        raise FileNotFoundError(f"Input label CSV not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    total_rows = len(df)

    filtered_df = df.copy()
    if args.start_date:
        filtered_df = filtered_df.loc[filtered_df["trade_date"].astype(str) >= args.start_date].copy()
    if args.end_date:
        filtered_df = filtered_df.loc[filtered_df["trade_date"].astype(str) <= args.end_date].copy()
    if args.symbols:
        symbols = {s.strip().upper() for s in _split_csv(args.symbols)}
        filtered_df = filtered_df.loc[filtered_df["ts_code"].astype(str).str.upper().isin(symbols)].copy()

    if output_path.exists():
        existing = pd.read_csv(output_path, low_memory=False)
        base = existing.copy() if len(existing) >= len(df) else df.copy()
    else:
        base = df.copy()

    for exit_time in exit_times:
        suf = _suffix(exit_time)
        if f"next_close_{suf}" not in base.columns:
            base[f"next_close_{suf}"] = pd.NA
        if f"overnight_return_{suf}" not in base.columns:
            base[f"overnight_return_{suf}"] = pd.NA
    if "minute_source" not in base.columns:
        base["minute_source"] = pd.NA
    if "minute_error" not in base.columns:
        base["minute_error"] = pd.NA
    if "minute_attempted_at" not in base.columns:
        base["minute_attempted_at"] = pd.NA

    already_complete_rows = 0
    cooldown_skipped_rows = 0
    attempted_rows = 0
    newly_completed_rows = 0
    updates: dict[tuple[str, str], dict] = {}

    candidate_rows = [pd.Series(row._asdict()) for row in base.itertuples(index=False)]
    candidate_rows = [
        r for r in candidate_rows
        if (not args.start_date or str(r.get("trade_date")) >= args.start_date)
        and (not args.end_date or str(r.get("trade_date")) <= args.end_date)
        and (not args.symbols or str(r.get("ts_code")).upper() in {s.strip().upper() for s in _split_csv(args.symbols)})
    ]
    if args.prefer_no_error_first:
        no_error_rows = [r for r in candidate_rows if not _has_minute_error(r)]
        error_rows = [r for r in candidate_rows if _has_minute_error(r)]
        if args.prioritize_recent:
            no_error_rows = sorted(no_error_rows, key=lambda r: (str(r.get("trade_date")), str(r.get("ts_code"))), reverse=True)
            error_rows = sorted(error_rows, key=lambda r: (str(r.get("trade_date")), str(r.get("ts_code"))), reverse=True)
        else:
            no_error_rows = sorted(no_error_rows, key=lambda r: (str(r.get("trade_date")), str(r.get("ts_code"))))
            error_rows = sorted(error_rows, key=lambda r: (str(r.get("trade_date")), str(r.get("ts_code"))))
        candidate_rows = no_error_rows + error_rows
    elif args.prioritize_recent:
        candidate_rows = sorted(candidate_rows, key=lambda r: (str(r.get("trade_date")), str(r.get("ts_code"))), reverse=True)

    for row_s in candidate_rows:
        key = (str(row_s.get("ts_code")), str(row_s.get("trade_date")))
        if _is_complete(row_s, exit_times):
            already_complete_rows += 1
            continue
        if _should_skip_due_to_cooldown(row_s, run_date=run_date, cooldown_days=args.cooldown_days):
            cooldown_skipped_rows += 1
            continue
        if args.batch_limit and attempted_rows >= args.batch_limit:
            continue
        enriched = enrich_overnight_row_with_intraday_exits(
            row_s.to_dict(),
            exit_times=exit_times,
            cache_dir=cache_dir,
            use_cache=True,
            write_cache=True,
            retries=args.retries,
            rate_limit_sleep_s=args.rate_limit_sleep,
            provider=args.minute_provider,
        )
        enriched["minute_attempted_at"] = datetime.now().isoformat(timespec="seconds")
        attempted_rows += 1
        minute_error = enriched.get("minute_error")
        if pd.isna(minute_error) or minute_error in (None, ""):
            newly_completed_rows += 1
        updates[key] = enriched

    out_rows = []
    for row in base.to_dict(orient="records"):
        key = (str(row.get("ts_code")), str(row.get("trade_date")))
        out_rows.append(updates.get(key, row))

    out = pd.DataFrame(out_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    minute_error_rows = int(out["minute_error"].map(_is_nonempty_error_value).sum()) if "minute_error" in out.columns else 0
    cache_files = len(list(cache_dir.glob("*.csv"))) if cache_dir.exists() else 0
    write_audit(
        audit_path,
        input_path=input_path,
        output_path=output_path,
        cache_dir=cache_dir,
        exit_times=exit_times,
        total_rows=total_rows,
        already_complete_rows=already_complete_rows,
        cooldown_skipped_rows=cooldown_skipped_rows,
        attempted_rows=attempted_rows,
        newly_completed_rows=newly_completed_rows,
        minute_error_rows=minute_error_rows,
        cache_files=cache_files,
        batch_limit=args.batch_limit,
        prioritize_recent=args.prioritize_recent,
        prefer_no_error_first=args.prefer_no_error_first,
        cooldown_days=args.cooldown_days,
        run_date=run_date,
    )

    print(f"Wrote enriched CSV: {output_path}")
    print(f"Wrote audit: {audit_path}")
    print(json.dumps({
        "total_rows_seen": total_rows,
        "already_complete_rows": already_complete_rows,
        "cooldown_skipped_rows": cooldown_skipped_rows,
        "attempted_rows": attempted_rows,
        "newly_completed_rows": newly_completed_rows,
        "minute_error_rows": minute_error_rows,
        "cache_files": cache_files,
        "prioritize_recent": args.prioritize_recent,
        "prefer_no_error_first": args.prefer_no_error_first,
        "cooldown_days": args.cooldown_days,
        "run_date": run_date,
        "minute_provider": args.minute_provider,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
