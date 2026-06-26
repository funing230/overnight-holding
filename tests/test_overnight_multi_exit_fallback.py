from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataflows import ashare_overnight_labels as labels


def test_normalize_intraday_frame_akshare_columns() -> None:
    raw = pd.DataFrame(
        {
            "时间": ["2026-01-06 09:35:00", "2026-01-06 09:40:00"],
            "开盘": [10.0, 10.1],
            "收盘": [10.05, 10.15],
            "最高": [10.1, 10.2],
            "最低": [9.9, 10.0],
            "成交量": [100, 120],
            "成交额": [1005, 1218],
        }
    )
    out = labels._normalize_intraday_frame(raw, "000001.SZ", "2026-01-06")
    assert list(out.columns) == ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
    assert len(out) == 2
    assert out.iloc[0]["ts_code"] == "000001.SZ"
    assert out.iloc[0]["close"] == 10.05


def test_get_intraday_bars_auto_falls_back_to_akshare(monkeypatch, tmp_path: Path) -> None:
    def fail_tushare(*args, **kwargs):
        raise RuntimeError("TushareRateLimitError: stk_mins 2次/天")

    def fake_akshare(ts_code: str, trade_date: str, freq: str = "5min") -> pd.DataFrame:
        return pd.DataFrame(
            {
                "trade_time": pd.to_datetime(["2026-01-06 09:35:00", "2026-01-06 09:45:00"]),
                "open": [10.0, 10.2],
                "high": [10.2, 10.4],
                "low": [9.9, 10.1],
                "close": [10.1, 10.3],
                "vol": [100, 150],
                "amount": [1010, 1545],
                "ts_code": [ts_code, ts_code],
            }
        )

    monkeypatch.setattr(labels, "_fetch_tushare_intraday_bars", fail_tushare)
    monkeypatch.setattr(labels, "_fetch_akshare_intraday_bars", fake_akshare)

    out = labels.get_intraday_bars("000001.SZ", "2026-01-06", cache_dir=tmp_path, provider="auto")
    assert len(out) == 2
    assert out.iloc[-1]["close"] == 10.3
    assert (tmp_path / "000001.SZ_20260106_5min_akshare.csv").exists()


def test_enrich_row_uses_auto_minute_provider(monkeypatch, tmp_path: Path) -> None:
    def fake_get_intraday_bars(*args, **kwargs):
        assert kwargs["provider"] == "auto"
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ", "000001.SZ"],
                "trade_time": pd.to_datetime(["2026-01-06 09:35:00", "2026-01-06 09:45:00", "2026-01-06 10:00:00"]),
                "close": [10.1, 10.2, 10.3],
            }
        )

    monkeypatch.setattr(labels, "get_intraday_bars", fake_get_intraday_bars)
    row = {"ts_code": "000001.SZ", "trade_date": "2026-01-05", "next_trade_date": "2026-01-06", "close": 10.0}
    out = labels.enrich_overnight_row_with_intraday_exits(row, cache_dir=tmp_path, provider="auto")
    assert out["minute_error"] is None
    assert out["next_close_0935"] == 10.1
    assert out["overnight_return_1000"] == 0.030000000000000027
