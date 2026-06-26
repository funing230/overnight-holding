from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_overnight_feature_table.py"
spec = importlib.util.spec_from_file_location("build_overnight_feature_table", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(mod)


def test_summarize_minute_features_builds_tail_session_metrics() -> None:
    mins = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 6,
            "trade_time": pd.to_datetime(
                [
                    "2026-04-01 14:30:00",
                    "2026-04-01 14:35:00",
                    "2026-04-01 14:40:00",
                    "2026-04-01 14:45:00",
                    "2026-04-01 14:50:00",
                    "2026-04-01 14:55:00",
                ]
            ),
            "open": [10.0, 10.1, 10.2, 10.25, 10.3, 10.4],
            "high": [10.1, 10.2, 10.3, 10.35, 10.5, 10.6],
            "low": [9.95, 10.0, 10.1, 10.2, 10.25, 10.35],
            "close": [10.05, 10.15, 10.25, 10.3, 10.45, 10.5],
            "vol": [100, 110, 120, 130, 140, 150],
            "amount": [1005, 1116.5, 1230, 1339, 1463, 1575],
        }
    )

    out = mod.summarize_minute_features(mins, "000001.SZ", "2026-04-01")
    assert out["ts_code"] == "000001.SZ"
    assert out["trade_date"] == "2026-04-01"
    assert out["minute_bar_count_30m"] == 6
    assert float(out["minute_last30_return"]) > 0
    assert float(out["minute_last15_return"]) > 0
    assert 0.0 <= float(out["minute_range_pos_30m"]) <= 1.0
    assert pd.notna(out["minute_vwap_gap_30m"])


def test_add_derived_features_computes_factor_and_minute_shares() -> None:
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 5,
            "trade_date": ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-07", "2026-04-08"],
            "close": [10.0, 10.2, 10.4, 10.3, 10.5],
            "next_open": [10.1, 10.25, 10.45, 10.35, 10.55],
            "overnight_return_open": [0.01, 0.005, 0.004, 0.003, 0.006],
            "gap_days": [1, 1, 1, 3, 1],
            "list_date": ["2020-01-01"] * 5,
            "is_limit_move_like": [False, False, False, False, False],
            "is_soft_outlier": [False, False, False, False, False],
            "total_mv": [1.0e6] * 5,
            "circ_mv": [8.0e5] * 5,
            "net_mf_amount": [1000.0, 1200.0, 1100.0, 1300.0, 900.0],
            "amount_t": [10000.0, 11000.0, 12000.0, 13000.0, 9000.0],
            "vol_t": [5000.0, 5200.0, 5100.0, 5300.0, 4900.0],
            "minute_vol_30m": [500.0, 510.0, 520.0, 530.0, 540.0],
            "minute_amount_30m": [900.0, 920.0, 940.0, 960.0, 980.0],
        }
    )

    out = mod.add_derived_features(df)
    assert "net_mf_ratio" in out.columns
    assert "minute_vol_share_30m" in out.columns
    assert "minute_amount_share_30m" in out.columns
    last = out.iloc[-1]
    assert pd.notna(last["log_total_mv"])
    assert pd.notna(last["net_mf_ratio"])
    assert pd.notna(last["minute_vol_share_30m"])
    assert pd.notna(last["minute_amount_share_30m"])
