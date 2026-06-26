import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_overnight_live_orchestrated as orch


def test_target_dt_and_seconds_until_are_ordered(monkeypatch):
    start = orch._target_dt("2026-05-21", "14:30:00")
    heavy_deadline = orch._target_dt("2026-05-21", "14:49:00")
    fallback_deadline = orch._target_dt("2026-05-21", "14:52:00")
    final_snapshot = orch._target_dt("2026-05-21", "14:54:30")
    final_fusion = orch._target_dt("2026-05-21", "14:55:30")
    top5_deadline = orch._target_dt("2026-05-21", "14:56:30")
    publish = orch._target_dt("2026-05-21", "14:57:00")
    assert start < heavy_deadline < fallback_deadline < final_snapshot < final_fusion < top5_deadline < publish
    assert (heavy_deadline - start).total_seconds() == 19 * 60
    assert (publish - final_fusion).total_seconds() == 90


def test_precheck_writes_status_file(tmp_path):
    class Args:
        out_root = str(tmp_path / "out")
        enable_minute_prefetch = True
        minute_prefetch_max_symbols = 20
        enable_ashare_enrichment = True
        ashare_enrichment_top_k = 30
        precheck_time = "14:00:00"
        prefilter_snapshot_time = "14:30:00"
        enhanced_deadline = "14:49:00"
        fallback_deadline = "14:52:00"
        final_snapshot_time = "14:54:30"
        final_fusion_time = "14:55:30"
        top5_deadline = "14:56:30"
        publish_deadline = "14:57:00"

    cwd = tmp_path / "repo"
    (cwd / "scripts").mkdir(parents=True)
    for name in ["fetch_realtime_snapshot.py", "run_overnight_live_multistage.py", "run_overnight_live_inference.py"]:
        (cwd / "scripts" / name).write_text("", encoding="utf-8")
    book = orch.RunBook(tmp_path / "run", "2026-05-21")
    orch._run_precheck(book, Args, cwd)
    status_path = book.run_dir / "precheck_status.json"
    payload = json.loads(status_path.read_text(encoding="utf-8"))
    assert payload["scripts"]["scripts/fetch_realtime_snapshot.py"] is True
    assert payload["deadlines"]["top5_deadline"] == "14:56:30"
    assert any(e["stage"] == "00_precheck" and e["status"] == "ok" for e in book.events)


def test_multistage_cmd_has_bounded_enhanced_and_fallback_modes(tmp_path):
    class Args:
        trade_date = "2026-05-21"
        heavy_top_k = 50
        heavy_target_top_n = 15
        light_top_k = 15
        final_top_n = 5
        final_candidate_pool_size = 50
        enable_minute_prefetch = True
        minute_prefetch_max_symbols = 20
        enable_ashare_enrichment = True
        ashare_enrichment_top_k = 30
        light_include_news_social_context = True
        dry_run_enhanced_heavy = False
        dry_run_enhanced_light = False

    pre = tmp_path / "pre.csv"
    final = tmp_path / "final.csv"
    enhanced = orch._multistage_cmd(Args, tmp_path / "enh", pre, final, fallback=False)
    assert "--enable-minute-prefetch" in enhanced
    assert "--minute-prefetch-max-symbols" in enhanced
    assert "20" in enhanced
    assert "--enable-ashare-enrichment" in enhanced
    assert "--ashare-enrichment-top-k" in enhanced
    assert "30" in enhanced
    assert "--dry-run-heavy" not in enhanced
    assert "--dry-run-light" not in enhanced

    fallback = orch._multistage_cmd(Args, tmp_path / "fb", pre, final, fallback=True)
    assert "--dry-run-heavy" in fallback
    assert "--dry-run-light" in fallback
    assert "--disable-social-hot-context" in fallback
    assert "--enable-minute-prefetch" not in fallback
    assert "--enable-ashare-enrichment" not in fallback


def test_runbook_report_updates_with_final_selected(tmp_path):
    run_dir = tmp_path / "live"
    book = orch.RunBook(run_dir, "2026-05-21")
    final = run_dir / "final.csv"
    pd.DataFrame(
        {
            "rank_in_live_day": [1, 2],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "name": ["A", "B"],
            "final_live_score": [0.8, 0.7],
            "quote_time": ["14:55:01", "14:55:02"],
        }
    ).to_csv(final, index=False)
    book.set_output("final_selected", str(final))
    report = (run_dir / "FULL_STEP_RESULTS.md").read_text(encoding="utf-8")
    assert "时间规划 / 硬截止" in report
    assert "最终 Top5" in report
    assert "000001.SZ" in report


def test_run_final_fusion_finds_nested_outputs(monkeypatch, tmp_path):
    book = orch.RunBook(tmp_path / "run", "2026-05-21")
    final_snapshot = tmp_path / "snapshot.csv"
    final_snapshot.write_text("ts_code,last_price\n000001.SZ,1\n", encoding="utf-8")
    enhanced_root = tmp_path / "enhanced"
    fallback_root = tmp_path / "fallback"
    nested = book.run_dir / "final_fusion_145530" / "2026-05-21" / "20260521_145530"
    nested.mkdir(parents=True)
    selected = nested / "live_selected_20260521_top5_pool50.csv"
    pd.DataFrame({"ts_code": ["000001.SZ"], "rank_in_live_day": [1]}).to_csv(selected, index=False)

    def fake_run_cmd(book_arg, stage, cmd, timeout_s=None, cwd=None):
        return subprocess.CompletedProcess(cmd, returncode=0)

    monkeypatch.setattr(orch, "_run_cmd", fake_run_cmd)

    class Args:
        trade_date = "2026-05-21"
        final_top_n = 5
        final_candidate_pool_size = 50
        final_fusion_timeout_seconds = 60

    found = orch._run_final_fusion(book, Args, final_snapshot, enhanced_root, fallback_root)
    assert found == selected
    status = json.loads((book.run_dir / "orchestrator_status.json").read_text(encoding="utf-8"))
    assert status["outputs"]["final_selected"] == str(selected)
