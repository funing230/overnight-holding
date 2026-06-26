from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from config.default_config import DEFAULT_CONFIG


@dataclass
class OvernightProviderConfig:
    feature_table_path: Path
    candidate_pool_size: int = 20
    top_n: int = 5
    filter_variant: str = "strict_risk"
    weight_variant: str = "baseline"
    exit_mode: str = "open"


class OvernightPipelineProvider:
    """Minimal graph-ready provider for overnight candidate access.

    This first version intentionally stays read-only and lightweight:
    - load the prebuilt feature table
    - fetch one trade date's candidate universe
    - expose compact summaries/payloads for agent graph consumption

    Later steps can extend this provider with baseline/strict-risk comparison,
    richer filtering logic, and backtest-ready portfolio outputs.
    """

    def __init__(self, config: OvernightProviderConfig | None = None):
        self.config = config or OvernightProviderConfig(
            feature_table_path=Path(DEFAULT_CONFIG["overnight_feature_table_path"]),
            candidate_pool_size=int(DEFAULT_CONFIG.get("overnight_candidate_pool_size", 20)),
            top_n=int(DEFAULT_CONFIG.get("overnight_top_n", 5)),
            filter_variant=str(DEFAULT_CONFIG.get("overnight_filter_variant", "strict_risk")),
            weight_variant=str(DEFAULT_CONFIG.get("overnight_weight_variant", "baseline")),
            exit_mode=str(DEFAULT_CONFIG.get("overnight_exit_mode", "open")),
        )
        self._df: pd.DataFrame | None = None

    def load_feature_table(self, force_reload: bool = False) -> pd.DataFrame:
        if self._df is not None and not force_reload:
            return self._df

        path = self.config.feature_table_path
        if not path.is_absolute():
            path = Path(DEFAULT_CONFIG["project_dir"]).parent / path
        if not path.exists():
            raise FileNotFoundError(f"Overnight feature table not found: {path}")

        df = pd.read_csv(path)
        if "trade_date" not in df.columns:
            raise ValueError("Feature table missing required column: trade_date")
        if "rank_in_day" not in df.columns:
            raise ValueError("Feature table missing required column: rank_in_day")
        if "overnight_score" not in df.columns:
            raise ValueError("Feature table missing required column: overnight_score")

        df = df.copy()
        df["trade_date"] = df["trade_date"].astype(str)
        df["rank_in_day"] = pd.to_numeric(df["rank_in_day"], errors="coerce")
        if "overnight_score" in df.columns:
            df["overnight_score"] = pd.to_numeric(df["overnight_score"], errors="coerce")
        self._df = df
        return df

    def get_trade_date_candidates(self, trade_date: str, top_k: int | None = None) -> pd.DataFrame:
        df = self.load_feature_table()
        k = top_k or self.config.candidate_pool_size
        day = df.loc[df["trade_date"] == str(trade_date)].copy()
        if day.empty:
            raise ValueError(f"No overnight candidates found for trade_date={trade_date}")

        day = day.sort_values(["rank_in_day", "overnight_score"], ascending=[True, False])
        day = day.loc[day["rank_in_day"].notna()].copy()
        day = day.loc[day["rank_in_day"] <= k].copy()
        return day.reset_index(drop=True)

    def summarize_trade_date_candidates(self, trade_date: str, top_k: int | None = None) -> dict[str, Any]:
        day = self.get_trade_date_candidates(trade_date, top_k=top_k)
        cols = [
            c for c in [
                "ts_code",
                "name",
                "industry",
                "rank_in_day",
                "overnight_score",
                "overnight_return_open",
                "close_vol_5d",
                "gap_days",
                "is_new_listing_180d",
                "is_limit_move_like",
                "is_soft_outlier",
                "is_extreme",
                "is_long_gap",
            ]
            if c in day.columns
        ]
        preview = day[cols].head(min(len(day), self.config.top_n)).to_dict(orient="records")
        return {
            "trade_date": str(trade_date),
            "candidate_count": int(len(day)),
            "top_n": int(self.config.top_n),
            "candidate_pool_size": int(top_k or self.config.candidate_pool_size),
            "filter_variant": self.config.filter_variant,
            "weight_variant": self.config.weight_variant,
            "exit_mode": self.config.exit_mode,
            "preview": preview,
        }

    def build_candidate_prompt_payload(self, trade_date: str, top_k: int | None = None) -> str:
        summary = self.summarize_trade_date_candidates(trade_date, top_k=top_k)
        lines = [
            f"trade_date: {summary['trade_date']}",
            f"filter_variant: {summary['filter_variant']}",
            f"weight_variant: {summary['weight_variant']}",
            f"exit_mode: {summary['exit_mode']}",
            f"candidate_count: {summary['candidate_count']}",
            f"candidate_pool_size: {summary['candidate_pool_size']}",
            f"top_n_target: {summary['top_n']}",
            "top_candidates:",
        ]
        for row in summary["preview"]:
            parts = [f"{k}={row[k]}" for k in row]
            lines.append("- " + ", ".join(parts))
        return "\n".join(lines)


def load_feature_table(path: str | Path | None = None, force_reload: bool = False) -> pd.DataFrame:
    provider = OvernightPipelineProvider()
    if path is not None:
        provider.config.feature_table_path = Path(path)
    return provider.load_feature_table(force_reload=force_reload)


def get_trade_date_candidates(trade_date: str, top_k: int | None = None) -> pd.DataFrame:
    return OvernightPipelineProvider().get_trade_date_candidates(trade_date, top_k=top_k)


def summarize_trade_date_candidates(trade_date: str, top_k: int | None = None) -> dict[str, Any]:
    return OvernightPipelineProvider().summarize_trade_date_candidates(trade_date, top_k=top_k)


def build_candidate_prompt_payload(trade_date: str, top_k: int | None = None) -> str:
    return OvernightPipelineProvider().build_candidate_prompt_payload(trade_date, top_k=top_k)
