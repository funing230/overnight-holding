#!/usr/bin/env python3
"""Summarize the latest deadline-driven overnight live run.

This helper is intentionally read-only.  It is meant for the 14:57 verification
step: do not trust chat delivery; inspect the TradingAgents2.0 artifact tree.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_ROOT = Path("data/overnight_live_actual")


def _latest_run(root: Path, trade_date: str) -> Path | None:
    date_dir = root / trade_date
    if not date_dir.exists():
        return None
    candidates = [p for p in date_dir.glob("live_*") if p.is_dir() and (p / "orchestrator_status.json").exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (p / "orchestrator_status.json").stat().st_mtime)[-1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_top(path: Path, n: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [
        c
        for c in [
            "rank_in_live_day",
            "rank_in_final_live_day",
            "ts_code",
            "name",
            "name_x",
            "industry",
            "final_live_score",
            "overnight_live_score",
            "heavy_score",
            "heavy_tier",
            "agent_score",
            "agent_risk_level",
            "last_price",
            "quote_time",
        ]
        if c in df.columns
    ]
    return df[cols].head(n)


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize latest overnight live orchestrated output")
    p.add_argument("--trade-date", required=True)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--run-dir", default=None, help="Specific run_dir; defaults to latest live_* under root/trade_date")
    p.add_argument("--top-n", type=int, default=5)
    p.add_argument("--markdown", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(Path(args.root), args.trade_date)
    if not run_dir:
        raise SystemExit(f"No orchestrated run found under {Path(args.root) / args.trade_date}")

    status_path = run_dir / "orchestrator_status.json"
    report_path = run_dir / "FULL_STEP_RESULTS.md"
    if not status_path.exists():
        raise SystemExit(f"Missing orchestrator_status.json: {status_path}")
    status = _load_json(status_path)
    outputs = status.get("outputs") or {}
    final_selected = outputs.get("final_selected")

    lines = [
        "# 一夜持股法实盘主动核验摘要",
        "",
        f"- trade_date: `{args.trade_date}`",
        f"- run_dir: `{run_dir}`",
        f"- status: `{status.get('status')}`",
        f"- updated_at: `{status.get('updated_at')}`",
        f"- quality_flags: `{', '.join(status.get('quality_flags') or []) if status.get('quality_flags') else 'none'}`",
        f"- orchestrator_status: `{status_path}`",
        f"- full_report: `{report_path}`",
        f"- final_selected: `{final_selected or 'MISSING'}`",
    ]
    if final_selected and Path(final_selected).exists():
        top = _read_top(Path(final_selected), args.top_n)
        lines.extend(["", f"## Top{args.top_n}", "", top.to_markdown(index=False)])
    else:
        lines.extend(["", "## TopN", "", "`final_selected` missing or file does not exist."])

    recent_events = (status.get("events") or [])[-8:]
    lines.extend(["", "## 最近事件", "", "| time | stage | status | reason |", "|---|---|---|---|"])
    for e in recent_events:
        lines.append(f"| {e.get('time','')} | {e.get('stage','')} | {e.get('status','')} | `{e.get('reason') or e.get('path') or e.get('cmd') or ''}` |")

    text = "\n".join(lines) + "\n"
    if args.markdown:
        print(text)
    else:
        print(f"run_dir={run_dir}")
        print(f"status={status.get('status')}")
        print(f"quality_flags={status.get('quality_flags') or []}")
        print(f"final_selected={final_selected or 'MISSING'}")
        if final_selected and Path(final_selected).exists():
            print(_read_top(Path(final_selected), args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
