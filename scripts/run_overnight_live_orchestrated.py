#!/usr/bin/env python3
"""Deadline-driven live orchestrator for the overnight holding workflow.

This wrapper is intentionally operational rather than research-oriented.  It
keeps the 14:30 -> 14:57 decision window safe by separating:

- a guaranteed fast/fallback path that must produce Top5 before the window; and
- an enhanced path (news/social/A-share/minute/LLM heavy+light) that may improve
  scores but is never allowed to block final snapshot/fusion.

Default Asia/Shanghai timetable:
- 14:00 precheck window (can be run earlier/later; records health only)
- 14:30 prefilter snapshot
- 14:31 enhanced multistage starts
- 14:49 enhanced path hard deadline (allows longer Heavy/Light than before)
- 14:52 fallback Top5/review materialization deadline
- 14:54:30 final snapshot
- 14:55:30 final fusion starts no matter what
- 14:56:30 final Top5 must be on disk
- 14:57:00 publish/report deadline

## Production usage

Recommended live command (schedule at 14:00 or earlier; it waits for gates):

```bash
python3 scripts/run_overnight_live_orchestrated.py \
  --trade-date YYYY-MM-DD \
  --out-root data/overnight_live_actual \
  --heavy-top-k 50 \
  --heavy-target-top-n 15 \
  --light-top-k 15 \
  --final-top-n 5 \
  --final-candidate-pool-size 50 \
  --minute-prefetch-max-symbols 20 \
  --ashare-enrichment-top-k 30
```

If the enhanced path is still running at 14:49, this orchestrator marks it
degraded and switches to fallback review materialization.  Final snapshot starts
at 14:54:30, final fusion starts at 14:55:30, and a report is always written to
`FULL_STEP_RESULTS.md`.

"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any

import pandas as pd


TZ_NAME = "Asia/Shanghai"
DEFAULT_OUT_ROOT = Path("data/overnight_live_actual")


def _now() -> datetime:
    return datetime.now().astimezone()


def _fmt_date(value: str) -> str:
    return value.replace("-", "")


def _parse_hms(value: str) -> dtime:
    parts = [int(x) for x in value.split(":")]
    if len(parts) == 2:
        parts.append(0)
    return dtime(parts[0], parts[1], parts[2])


def _target_dt(trade_date: str, hms: str) -> datetime:
    y, m, d = [int(x) for x in trade_date.split("-")]
    t = _parse_hms(hms)
    # Local system for this workspace is configured as Asia/Shanghai during live runs.
    return datetime(y, m, d, t.hour, t.minute, t.second).astimezone()


def _seconds_until(trade_date: str, hms: str) -> float:
    return (_target_dt(trade_date, hms) - _now()).total_seconds()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


class RunBook:
    def __init__(self, run_dir: Path, trade_date: str):
        self.run_dir = run_dir
        self.trade_date = trade_date
        self.status_path = run_dir / "orchestrator_status.json"
        self.report_path = run_dir / "FULL_STEP_RESULTS.md"
        self.events: list[dict[str, Any]] = []
        self.quality_flags: list[str] = []
        self.outputs: dict[str, Any] = {}
        self.status = "running"
        _ensure_dir(run_dir / "logs")
        self.flush()

    def event(self, stage: str, status: str, **kw: Any) -> None:
        item = {
            "time": _now().isoformat(timespec="seconds"),
            "stage": stage,
            "status": status,
            **kw,
        }
        self.events.append(item)
        if status in {"degraded", "timeout", "failed", "skipped"}:
            reason = kw.get("reason") or stage
            flag = f"{stage}:{status}:{reason}"
            if flag not in self.quality_flags:
                self.quality_flags.append(flag)
        self.flush()

    def set_output(self, key: str, value: Any) -> None:
        self.outputs[key] = value
        self.flush()

    def flush(self) -> None:
        _write_json(self.status_path, {
            "trade_date": self.trade_date,
            "run_dir": str(self.run_dir),
            "status": self.status,
            "updated_at": _now().isoformat(timespec="seconds"),
            "quality_flags": self.quality_flags,
            "outputs": self.outputs,
            "events": self.events,
        })
        self._write_report()

    def _write_report(self) -> None:
        lines = [
            "# 一夜持股法实盘流程结果",
            "",
            f"- trade_date: `{self.trade_date}`",
            f"- run_dir: `{self.run_dir}`",
            f"- status: `{self.status}`",
            f"- updated_at: `{_now().isoformat(timespec='seconds')}`",
            f"- quality_flags: `{', '.join(self.quality_flags) if self.quality_flags else 'none'}`",
            "",
            "## 时间规划 / 硬截止",
            "",
            "| 时间 | 任务 | 硬规则 |",
            "|---|---|---|",
            "| 14:00 | precheck 数据源/模型/目录 | 只记录健康状态，不阻塞主链路 |",
            "| 14:30 | prefilter snapshot + Recall Top50 | snapshot 失败才终止；增强项不得阻塞 |",
            "| 14:31-14:49 | enhanced path：minute/news/social/A-share/Heavy/Light | 允许重度/轻度分析延长，但 14:49 必须结束或杀掉 |",
            "| 14:49-14:52 | fallback path | 若增强未完成，必须产出中性 review 可用文件 |",
            "| 14:54:30 | final snapshot | 新 snapshot 失败则用最近可用 snapshot |",
            "| 14:55:30 | final fusion | 无论增强是否完成都开始 final |",
            "| 14:56:30 | Top5 落盘 | 未落盘即 degraded/failed |",
            "| 14:57:00 | 汇报 | 必须报告 Top5 或明确失败原因 |",
            "",
            "## 事件",
            "",
            "| 时间 | 阶段 | 状态 | 说明 |",
            "|---|---|---|---|",
        ]
        for e in self.events:
            detail = e.get("reason") or e.get("path") or e.get("cmd") or ""
            lines.append(f"| {e.get('time','')} | {e.get('stage','')} | {e.get('status','')} | `{detail}` |")
        lines.extend(["", "## 输出", ""])
        for k, v in self.outputs.items():
            lines.append(f"- {k}: `{v}`")
        final_selected = self.outputs.get("final_selected")
        if final_selected and Path(final_selected).exists():
            try:
                df = pd.read_csv(final_selected)
                cols = [c for c in ["rank_in_live_day", "ts_code", "name", "name_x", "industry", "final_live_score", "overnight_live_score", "heavy_score", "heavy_tier", "agent_score", "agent_risk_level", "last_price", "quote_time"] if c in df.columns]
                lines.extend(["", "## 最终 Top5", "", df[cols].head(5).to_markdown(index=False)])
            except Exception as exc:
                lines.append(f"\n无法读取最终 Top5: `{exc}`")
        self.report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_cmd(book: RunBook, stage: str, cmd: list[str], *, timeout_s: int | None = None, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    logs = book.run_dir / "logs"
    stdout_path = logs / f"{stage}.stdout.log"
    stderr_path = logs / f"{stage}.stderr.log"
    book.event(stage, "start", cmd=" ".join(cmd), timeout_s=timeout_s)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        try:
            proc = subprocess.run(cmd, cwd=str(cwd or Path.cwd()), stdout=out, stderr=err, text=True, timeout=timeout_s)
            elapsed = round(time.time() - started, 3)
            status = "ok" if proc.returncode == 0 else "failed"
            book.event(stage, status, rc=proc.returncode, elapsed_seconds=elapsed, stdout=str(stdout_path), stderr=str(stderr_path))
            return proc
        except subprocess.TimeoutExpired:
            elapsed = round(time.time() - started, 3)
            book.event(stage, "timeout", reason=f"timeout_after_{timeout_s}s", elapsed_seconds=elapsed, stdout=str(stdout_path), stderr=str(stderr_path))
            return subprocess.CompletedProcess(cmd, returncode=124)


def _wait_until(book: RunBook, trade_date: str, hms: str, label: str, *, no_wait: bool = False) -> None:
    remaining = _seconds_until(trade_date, hms)
    if remaining <= 0 or no_wait:
        book.event(label, "skipped" if remaining <= 0 else "ok", reason="target_time_already_passed" if remaining <= 0 else "no_wait")
        return
    book.event(label, "waiting", reason=f"wait_until_{hms}", seconds=round(remaining, 1))
    time.sleep(remaining)
    book.event(label, "ok", reason=f"reached_{hms}")


def _deadline_status(book: RunBook, trade_date: str, hms: str, label: str, *, ok_reason: str) -> None:
    remaining = _seconds_until(trade_date, hms)
    if remaining >= 0:
        book.event(label, "ok", reason=ok_reason, seconds_before_deadline=round(remaining, 1))
    else:
        book.event(label, "degraded", reason=f"deadline_missed_{hms}", seconds_late=round(abs(remaining), 1))


def _run_precheck(book: RunBook, args: argparse.Namespace, cwd: Path) -> None:
    checks = {
        "cwd": str(cwd),
        "scripts": {},
        "out_root_parent_exists": Path(args.out_root).parent.exists(),
        "enhancements": {
            "minute_prefetch_enabled": bool(args.enable_minute_prefetch),
            "minute_prefetch_max_symbols": int(args.minute_prefetch_max_symbols),
            "ashare_enrichment_enabled": bool(args.enable_ashare_enrichment),
            "ashare_enrichment_top_k": int(args.ashare_enrichment_top_k),
        },
        "deadlines": {
            "precheck_time": args.precheck_time,
            "prefilter_snapshot_time": args.prefilter_snapshot_time,
            "enhanced_deadline": args.enhanced_deadline,
            "fallback_deadline": args.fallback_deadline,
            "final_snapshot_time": args.final_snapshot_time,
            "final_fusion_time": args.final_fusion_time,
            "top5_deadline": args.top5_deadline,
            "publish_deadline": args.publish_deadline,
        },
    }
    required_scripts = [
        "scripts/fetch_realtime_snapshot.py",
        "scripts/run_overnight_live_multistage.py",
        "scripts/run_overnight_live_inference.py",
    ]
    missing = []
    for script in required_scripts:
        exists = (cwd / script).exists()
        checks["scripts"][script] = exists
        if not exists:
            missing.append(script)
    precheck_path = book.run_dir / "precheck_status.json"
    _write_json(precheck_path, checks)
    book.set_output("precheck_status", str(precheck_path))
    if missing:
        book.event("00_precheck", "failed", reason=f"missing_scripts:{','.join(missing)}")
    else:
        book.event("00_precheck", "ok", path=str(precheck_path))


def _find_latest(pattern_root: Path, glob_pat: str) -> Path | None:
    files = sorted(pattern_root.glob(glob_pat), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    return files[-1] if files else None


def _snapshot_cmd(args: argparse.Namespace, out: Path, manifest: Path, min_quote_time: str | None) -> list[str]:
    cmd = [
        "python3", "scripts/fetch_realtime_snapshot.py",
        "--trade-date", args.trade_date,
        "--source", args.snapshot_source,
        "--fallback-source", args.snapshot_fallback_source,
        "--min-coverage", str(args.min_coverage),
        "--out", str(out),
        "--manifest", str(manifest),
    ]
    if min_quote_time:
        cmd += ["--min-quote-time", min_quote_time]
    return cmd


def _multistage_cmd(args: argparse.Namespace, out_root: Path, pre_snapshot: Path, final_snapshot: Path, *, fallback: bool) -> list[str]:
    cmd = [
        "python3", "scripts/run_overnight_live_multistage.py",
        "--trade-date", args.trade_date,
        "--prefilter-snapshot-csv", str(pre_snapshot),
        "--final-snapshot-csv", str(final_snapshot),
        "--out-root", str(out_root),
        "--heavy-top-k", str(args.heavy_top_k),
        "--heavy-target-top-n", str(args.heavy_target_top_n),
        "--light-top-k", str(args.light_top_k),
        "--final-top-n", str(args.final_top_n),
        "--final-candidate-pool-size", str(args.final_candidate_pool_size),
    ]
    if fallback:
        cmd += ["--dry-run-heavy", "--dry-run-light", "--disable-social-hot-context", "--disable-xueqiu", "--disable-twitter"]
    else:
        if args.enable_minute_prefetch:
            cmd += ["--enable-minute-prefetch", "--minute-prefetch-missing-only"]
            if args.minute_prefetch_max_symbols:
                cmd += ["--minute-prefetch-max-symbols", str(args.minute_prefetch_max_symbols)]
        if args.enable_ashare_enrichment:
            cmd += ["--enable-ashare-enrichment", "--ashare-enrichment-top-k", str(args.ashare_enrichment_top_k)]
        if args.light_include_news_social_context:
            cmd.append("--light-include-news-social-context")
        if getattr(args, "dry_run_enhanced_heavy", False):
            cmd.append("--dry-run-heavy")
        if getattr(args, "dry_run_enhanced_light", False):
            cmd.append("--dry-run-light")
    return cmd


def _write_final_from_multistage(book: RunBook, run_root: Path, label: str) -> Path | None:
    selected = _find_latest(run_root, "*/**/04_final_fusion/live_selected_*final_top*.csv")
    if selected:
        book.set_output(f"{label}_selected", str(selected))
        manifest = _find_latest(run_root, "*/**/multistage_manifest.json")
        if manifest:
            book.set_output(f"{label}_manifest", str(manifest))
        return selected
    return None


def _run_final_fusion(book: RunBook, args: argparse.Namespace, final_snapshot: Path, enhanced_root: Path, fallback_root: Path) -> Path | None:
    enhanced_manifest = _find_latest(enhanced_root, "*/**/multistage_manifest.json")
    fallback_manifest = _find_latest(fallback_root, "*/**/multistage_manifest.json")
    source_manifest = enhanced_manifest if enhanced_manifest and enhanced_manifest.exists() else fallback_manifest
    heavy_scores = light_scores = None
    if source_manifest:
        try:
            m = _read_json(source_manifest)
            heavy_scores = (((m.get("stages") or {}).get("selector") or {}).get("paths") or {}).get("selector_review_scores")
            light_scores = (((m.get("stages") or {}).get("scorer") or {}).get("paths") or {}).get("scorer_review_scores")
            book.set_output("review_source_manifest", str(source_manifest))
        except Exception as exc:
            book.event("resolve_review_scores", "degraded", reason=str(exc))
    out_root = book.run_dir / "final_fusion_145530"
    cmd = [
        "python3", "scripts/run_overnight_live_inference.py",
        "--trade-date", args.trade_date,
        "--snapshot-csv", str(final_snapshot),
        "--top-n", str(args.final_top_n),
        "--candidate-pool-size", str(args.final_candidate_pool_size),
        "--out-root", str(out_root),
    ]
    if heavy_scores and Path(heavy_scores).exists():
        cmd += ["--heavy-review-scores", heavy_scores]
    else:
        book.event("heavy_scores", "degraded", reason="missing_use_deterministic_only")
    if light_scores and Path(light_scores).exists():
        cmd += ["--light-review-scores", light_scores]
    else:
        book.event("light_scores", "degraded", reason="missing_use_deterministic_only")
    proc = _run_cmd(book, "05_final_fusion", cmd, timeout_s=args.final_fusion_timeout_seconds)
    selected = _find_latest(out_root, "*/*/live_selected_*top*.csv")
    if proc.returncode == 0 and selected:
        book.set_output("final_selected", str(selected))
        return selected
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Deadline-driven 14:30->14:57 overnight live orchestrator")
    p.add_argument("--trade-date", required=True)
    p.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    p.add_argument("--no-wait", action="store_true", help="Do not sleep for wall-clock gates; useful for tests/manual reruns")
    p.add_argument("--snapshot-source", default="auto", choices=["auto", "tencent", "tushare"])
    p.add_argument("--snapshot-fallback-source", default="tushare", choices=["tushare", "tencent", "none"])
    p.add_argument("--min-coverage", type=float, default=0.80)
    p.add_argument("--precheck-time", default="14:00:00")
    p.add_argument("--prefilter-snapshot-time", default="14:30:00")
    p.add_argument("--enhanced-deadline", default="14:49:00", help="Hard deadline for enhanced minute/news/A-share/Heavy/Light path")
    p.add_argument("--fallback-deadline", default="14:52:00")
    p.add_argument("--final-snapshot-time", default="14:54:30")
    p.add_argument("--final-fusion-time", default="14:55:30")
    p.add_argument("--top5-deadline", default="14:56:30")
    p.add_argument("--publish-deadline", default="14:57:00")
    p.add_argument("--heavy-top-k", type=int, default=50)
    p.add_argument("--heavy-target-top-n", type=int, default=15)
    p.add_argument("--light-top-k", type=int, default=15)
    p.add_argument("--final-top-n", type=int, default=5)
    p.add_argument("--final-candidate-pool-size", type=int, default=50)
    p.add_argument("--enable-minute-prefetch", action="store_true", default=True)
    p.add_argument("--disable-minute-prefetch", action="store_true", help="Disable opportunistic minute prefetch even on enhanced path")
    p.add_argument("--minute-prefetch-max-symbols", type=int, default=20, help="Bound minute requests so it cannot exhaust the live window")
    p.add_argument("--enable-ashare-enrichment", action="store_true", default=True)
    p.add_argument("--disable-ashare-enrichment", action="store_true", help="Disable A-share enrichment even on enhanced path")
    p.add_argument("--ashare-enrichment-top-k", type=int, default=30)
    p.add_argument("--light-include-news-social-context", action="store_true")
    p.add_argument("--dry-run-enhanced-heavy", action="store_true", help="Use neutral heavy scores on enhanced path; mainly for smoke tests")
    p.add_argument("--dry-run-enhanced-light", action="store_true", help="Use neutral light scores on enhanced path; mainly for smoke tests")
    p.add_argument("--fallback-timeout-seconds", type=int, default=150)
    p.add_argument("--final-snapshot-timeout-seconds", type=int, default=45)
    p.add_argument("--final-fusion-timeout-seconds", type=int, default=60)
    args = p.parse_args()
    if args.disable_minute_prefetch:
        args.enable_minute_prefetch = False
    if args.disable_ashare_enrichment:
        args.enable_ashare_enrichment = False

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / args.trade_date / f"live_{_fmt_date(args.trade_date)}_{run_ts}"
    _ensure_dir(run_dir)
    book = RunBook(run_dir, args.trade_date)
    book.event("plan", "ok", reason="deadline_driven_fast_and_enhanced_paths")

    cwd = Path.cwd()
    _wait_until(book, args.trade_date, args.precheck_time, "00_wait_precheck", no_wait=args.no_wait)
    _run_precheck(book, args, cwd)
    if any(e.get("stage") == "00_precheck" and e.get("status") == "failed" for e in book.events):
        book.status = "failed"
        book.flush()
        raise SystemExit(2)

    snapshots = run_dir / "snapshots"
    _ensure_dir(snapshots)

    _wait_until(book, args.trade_date, args.prefilter_snapshot_time, "00_wait_prefilter_snapshot", no_wait=args.no_wait)
    pre_snapshot = snapshots / f"snapshot_{_fmt_date(args.trade_date)}_1430_auto.csv"
    pre_manifest = snapshots / f"snapshot_{_fmt_date(args.trade_date)}_1430_auto.manifest.json"
    proc = _run_cmd(book, "01_fetch_prefilter_snapshot", _snapshot_cmd(args, pre_snapshot, pre_manifest, None), timeout_s=45, cwd=cwd)
    if proc.returncode != 0 or not pre_snapshot.exists():
        book.status = "failed"
        book.event("abort", "failed", reason="prefilter_snapshot_failed")
        book.flush()
        raise SystemExit(2)
    book.set_output("prefilter_snapshot", str(pre_snapshot))
    book.set_output("prefilter_manifest", str(pre_manifest))

    # Enhanced path gets the larger budget, but cannot cross the hard deadline.
    enhanced_root = run_dir / "multistage_work_enhanced"
    enhanced_budget = max(1, int(_seconds_until(args.trade_date, args.enhanced_deadline))) if not args.no_wait else max(1, int((_parse_hms(args.enhanced_deadline).hour * 3600 + _parse_hms(args.enhanced_deadline).minute * 60 + _parse_hms(args.enhanced_deadline).second) - (_parse_hms(args.prefilter_snapshot_time).hour * 3600 + _parse_hms(args.prefilter_snapshot_time).minute * 60 + _parse_hms(args.prefilter_snapshot_time).second)))
    proc = _run_cmd(book, "02_enhanced_multistage", _multistage_cmd(args, enhanced_root, pre_snapshot, pre_snapshot, fallback=False), timeout_s=enhanced_budget, cwd=cwd)
    enhanced_selected = _write_final_from_multistage(book, enhanced_root, "enhanced")
    if proc.returncode != 0 or not enhanced_selected:
        book.event("02_enhanced_multistage", "degraded", reason="enhanced_unavailable_before_deadline")

    fallback_root = run_dir / "multistage_work_fallback"
    fallback_needed = not enhanced_selected
    if fallback_needed:
        fallback_budget = min(args.fallback_timeout_seconds, max(1, int(_seconds_until(args.trade_date, args.fallback_deadline))) if not args.no_wait else args.fallback_timeout_seconds)
        proc = _run_cmd(book, "03_fallback_multistage", _multistage_cmd(args, fallback_root, pre_snapshot, pre_snapshot, fallback=True), timeout_s=fallback_budget, cwd=cwd)
        fallback_selected = _write_final_from_multistage(book, fallback_root, "fallback")
        if proc.returncode != 0 or not fallback_selected:
            book.event("03_fallback_multistage", "failed", reason="fallback_failed_before_final_window")
    else:
        book.event("03_fallback_multistage", "skipped", reason="enhanced_available")

    _wait_until(book, args.trade_date, args.final_snapshot_time, "04_wait_final_snapshot", no_wait=args.no_wait)
    final_snapshot = snapshots / f"snapshot_{_fmt_date(args.trade_date)}_1455_auto.csv"
    final_manifest = snapshots / f"snapshot_{_fmt_date(args.trade_date)}_1455_auto.manifest.json"
    proc = _run_cmd(book, "04_fetch_final_snapshot", _snapshot_cmd(args, final_snapshot, final_manifest, "14:54:00"), timeout_s=args.final_snapshot_timeout_seconds, cwd=cwd)
    if proc.returncode != 0 or not final_snapshot.exists():
        book.event("04_fetch_final_snapshot", "degraded", reason="use_prefilter_snapshot_fallback")
        final_snapshot = pre_snapshot
        final_manifest = pre_manifest
    book.set_output("final_snapshot", str(final_snapshot))
    book.set_output("final_manifest", str(final_manifest))

    _wait_until(book, args.trade_date, args.final_fusion_time, "05_wait_final_fusion", no_wait=args.no_wait)
    selected = _run_final_fusion(book, args, final_snapshot, enhanced_root, fallback_root)
    if not selected:
        # Last resort: if a multistage final exists, publish it rather than miss the window.
        selected = enhanced_selected or _find_latest(fallback_root, "*/**/04_final_fusion/live_selected_*final_top*.csv")
        if selected:
            book.event("05_final_fusion", "degraded", reason="using_existing_multistage_final")
            book.set_output("final_selected", str(selected))
    if selected:
        _deadline_status(book, args.trade_date, args.top5_deadline, "06_top5_deadline", ok_reason="top5_output_ready")
        book.status = "completed" if not book.quality_flags else "degraded_completed"
    else:
        book.status = "failed"
        book.event("final_selected", "failed", reason="no_top5_output")
    _deadline_status(book, args.trade_date, args.publish_deadline, "07_publish_deadline", ok_reason="report_ready")
    book.flush()

    print(f"Run dir: {run_dir}")
    print(f"Status: {book.status}")
    print(f"Report: {book.report_path}")
    if selected:
        print(f"Final selected: {selected}")


if __name__ == "__main__":
    main()
