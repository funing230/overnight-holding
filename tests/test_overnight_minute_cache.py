from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataflows.overnight_minute_cache import (
    load_cached_minute_window_frame,
    minute_cache_path,
    summarize_minute_features,
)


def test_minute_cache_roundtrip(tmp_path: Path) -> None:
    cache_dir = tmp_path / "minute_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = minute_cache_path(cache_dir, "000001.SZ", "2026-05-14", "5min", "14:30:00", "15:00:00")
    df = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_time": ["2026-05-14 14:30:00", "2026-05-14 14:35:00"],
            "open": [10.0, 10.1],
            "high": [10.1, 10.2],
            "low": [9.9, 10.0],
            "close": [10.05, 10.15],
            "vol": [100, 120],
            "amount": [1005, 1218],
        }
    )
    df.to_csv(path, index=False)

    loaded, err = load_cached_minute_window_frame(
        "000001.SZ", "2026-05-14", cache_dir, start_time="14:30:00", end_time="15:00:00", freq="5min"
    )
    assert err is None
    assert len(loaded) == 2
    assert pd.api.types.is_datetime64_any_dtype(loaded["trade_time"])


def test_summarize_minute_features_empty_cache_miss_shape() -> None:
    out = summarize_minute_features(pd.DataFrame(), "000001.SZ", "2026-05-14")
    assert out["ts_code"] == "000001.SZ"
    assert out["trade_date"] == "2026-05-14"
    assert out["minute_bar_count_30m"] == 0
    assert pd.isna(out["minute_last30_return"])
