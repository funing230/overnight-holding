import pandas as pd

from dataflows.overnight_live_selector_review_provider import (
    parse_selector_review_scores,
    apply_selector_selection,
)
from dataflows.overnight_live_provider import apply_multi_stage_review_fusion


def _candidate_pool():
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ", "000006.SZ"],
            "overnight_live_score": [0.90, 0.85, 0.80, 0.75, 0.70, 0.65],
            "live_pass_risk_filter": [True] * 6,
            "live_reject_reasons": [""] * 6,
        }
    )


def test_selector_new_schema_materializes_default_watch_for_unmentioned_candidates():
    pool = _candidate_pool()
    text = """SELECTOR_REVIEW_JSON_START
{
  "trade_date": "2026-05-14",
  "target_top_n": 3,
  "top_picks": [
    {
      "ts_code": "000004.SZ",
      "heavy_score": 0.93,
      "heavy_tier": "core",
      "heavy_veto": false,
      "heavy_adjustment": 0.08,
      "heavy_keep_rank": 1,
      "heavy_reason": "best",
      "heavy_risk_flags": []
    },
    {
      "ts_code": "000003.SZ",
      "heavy_score": 0.81,
      "heavy_tier": "watch",
      "heavy_veto": false,
      "heavy_adjustment": 0.02,
      "heavy_keep_rank": 2,
      "heavy_reason": "ok",
      "heavy_risk_flags": []
    }
  ],
  "rejects": [
    {
      "ts_code": "000006.SZ",
      "heavy_score": 0.15,
      "heavy_tier": "reject",
      "heavy_veto": true,
      "heavy_adjustment": -0.2,
      "heavy_keep_rank": 999,
      "heavy_reason": "bad",
      "heavy_risk_flags": ["risk"]
    }
  ],
  "summary": {
    "core_count": 1,
    "watch_top15_count": 1,
    "reject_count": 1,
    "notes": "others default watch"
  }
}
SELECTOR_REVIEW_JSON_END"""
    result = parse_selector_review_scores(text, pool)
    assert result.parsed is True
    assert result.error == ""
    by_code = result.scores.set_index("ts_code")
    assert by_code.loc["000004.SZ", "heavy_tier"] == "core"
    assert by_code.loc["000003.SZ", "heavy_keep_rank"] == 2
    assert bool(by_code.loc["000006.SZ", "heavy_veto"]) is True
    assert by_code.loc["000001.SZ", "heavy_reason"] == "default_watch_unmentioned"
    assert by_code.loc["000001.SZ", "heavy_tier"] == "watch"
    assert by_code.loc["000001.SZ", "heavy_keep_rank"] >= 100


def test_selector_json_recovery_without_end_marker_uses_structured_reviews():
    pool = _candidate_pool()
    text = """SELECTOR_REVIEW_JSON_START
{
  "trade_date": "2026-05-14",
  "target_top_n": 3,
  "top_picks": [
    {
      "ts_code": "000001.SZ",
      "heavy_score": 0.91,
      "heavy_tier": "core",
      "heavy_veto": false,
      "heavy_adjustment": 0.05,
      "heavy_keep_rank": 1,
      "heavy_reason": "strong",
      "heavy_risk_flags": []
    },
    {
      "ts_code": "000002.SZ",
      "heavy_score": 0.81,
      "heavy_tier": "watch",
      "heavy_veto": false,
      "heavy_adjustment": 0.01,
      "heavy_keep_rank": 2,
      "heavy_reason": "ok",
      "heavy_risk_flags": []
    }
  ],
  "rejects": [
    {
      "ts_code": "000003.SZ",
      "heavy_score": 0.21,
      "heavy_tier": "reject",
      "heavy_veto": true,
      "heavy_adjustment": -0.2,
      "heavy_keep_rank": 999,
      "heavy_reason": "bad",
      "heavy_risk_flags": []
    }
  ],
  "summary": {
    "core_count": 1,
    "watch_top15_count": 1,
    "reject_count": 1,
    "notes": "000001 最优"
  }
}
简短总结：000001 最优。"""
    result = parse_selector_review_scores(text, pool)
    assert result.parsed is True
    assert result.error == ""
    by_code = result.scores.set_index("ts_code")
    assert by_code.loc["000001.SZ", "heavy_score"] == 0.91
    assert by_code.loc["000002.SZ", "heavy_tier"] == "watch"
    assert bool(by_code.loc["000003.SZ", "heavy_veto"]) is True


def test_selector_ranked_text_fallback_prefers_last_final_list():
    pool = _candidate_pool()
    text = """
1. 000001.SZ first evaluation order
2. 000002.SZ first evaluation order
3. 000003.SZ first evaluation order
4. 000004.SZ first evaluation order
5. 000005.SZ first evaluation order
6. 000006.SZ first evaluation order

Final Top15:
1. 000004.SZ final best
2. 000003.SZ final second
3. 000001.SZ final third
4. 000006.SZ final fourth
5. 000002.SZ final fifth
"""
    result = parse_selector_review_scores(text, pool)
    assert result.parsed is True
    assert "ranked_text_fallback" in result.error
    ranked = result.scores.sort_values("heavy_keep_rank")
    assert ranked.iloc[0]["ts_code"] == "000004.SZ"
    assert ranked.iloc[1]["ts_code"] == "000003.SZ"
    selected = apply_selector_selection(pool, result.scores, target_top_n=3)
    assert set(selected["ts_code"]).issubset({"000004.SZ", "000003.SZ", "000001.SZ", "000006.SZ", "000002.SZ"})


def test_multi_stage_fusion_applies_selector_and_scorer_veto(tmp_path):
    scored = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "overnight_live_score": [0.95, 0.90, 0.60],
            "live_pass_risk_filter": [True, True, True],
            "live_reject_reasons": ["", "", ""],
        }
    )
    heavy = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "heavy_score": [0.9, 0.9, 0.9],
            "heavy_tier": ["reject", "core", "core"],
            "heavy_veto": [False, False, False],
            "heavy_adjustment": [0.0, 0.0, 0.0],
            "heavy_keep_rank": [1, 2, 3],
            "heavy_reason": ["reject by heavy", "ok", "ok"],
            "heavy_risk_flags": ["[]", "[]", "[]"],
        }
    )
    light = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "agent_score": [0.9, 0.9, 0.9],
            "agent_risk_level": ["low", "low", "low"],
            "agent_veto": [False, True, False],
            "agent_adjustment": [0.0, 0.0, 0.0],
            "agent_reason": ["ok", "light veto", "ok"],
        }
    )
    selector_path = tmp_path / "heavy.csv"
    scorer_path = tmp_path / "light.csv"
    heavy.to_csv(selector_path, index=False)
    light.to_csv(scorer_path, index=False)

    fused = apply_multi_stage_review_fusion(scored, selector_path, scorer_path)
    by_code = fused.set_index("ts_code")
    assert by_code.loc["000001.SZ", "live_pass_risk_filter"] is False or by_code.loc["000001.SZ", "live_pass_risk_filter"] == False
    assert by_code.loc["000002.SZ", "live_pass_risk_filter"] is False or by_code.loc["000002.SZ", "live_pass_risk_filter"] == False
    assert by_code.loc["000003.SZ", "live_pass_risk_filter"] is True or by_code.loc["000003.SZ", "live_pass_risk_filter"] == True
    assert "heavy_reject" in by_code.loc["000001.SZ", "live_reject_reasons"]
    assert "agent_veto" in by_code.loc["000002.SZ", "live_reject_reasons"]
    assert by_code.loc["000001.SZ", "final_live_score"] == -999.0
    assert by_code.loc["000002.SZ", "final_live_score"] == -999.0


def test_multi_stage_fusion_adds_social_bonus_score(tmp_path):
    scored = pd.DataFrame(
        {
            "ts_code": ["000333.SZ", "600941.SH"],
            "overnight_live_score": [0.80, 0.80],
            "live_pass_risk_filter": [True, True],
            "live_reject_reasons": ["", ""],
        }
    )
    heavy = pd.DataFrame(
        {
            "ts_code": ["000333.SZ", "600941.SH"],
            "heavy_score": [0.5, 0.5],
            "heavy_tier": ["core", "core"],
            "heavy_veto": [False, False],
            "heavy_adjustment": [0.0, 0.0],
            "heavy_keep_rank": [1, 2],
            "heavy_reason": ["ok", "ok"],
            "heavy_risk_flags": ["[]", "[]"],
        }
    )
    light = pd.DataFrame(
        {
            "ts_code": ["000333.SZ", "600941.SH"],
            "agent_score": [0.5, 0.5],
            "agent_risk_level": ["low", "low"],
            "agent_veto": [False, False],
            "agent_adjustment": [0.0, 0.0],
            "agent_reason": ["ok", "ok"],
        }
    )
    social = pd.DataFrame(
        {
            "ts_code": ["000333.SZ", "600941.SH"],
            "hot_mention_count": [3, 0],
            "hot_source_count": [2, 0],
            "hot_best_rank": [3, None],
            "hot_recency_hours": [1.0, None],
            "social_bonus_score": [0.04, 0.0],
        }
    )
    theme = pd.DataFrame(
        {
            "ts_code": ["000333.SZ", "600941.SH"],
            "theme_match_count": [2, 0],
            "theme_source_count": [2, 0],
            "theme_names": ["AI/终端智能化|宏观外交/风险偏好", ""],
            "theme_bonus_score": [0.05, 0.0],
        }
    )
    selector_path = tmp_path / "heavy.csv"
    scorer_path = tmp_path / "light.csv"
    social_path = tmp_path / "social.csv"
    theme_path = tmp_path / "theme.csv"
    heavy.to_csv(selector_path, index=False)
    light.to_csv(scorer_path, index=False)
    social.to_csv(social_path, index=False)
    theme.to_csv(theme_path, index=False)

    fused = apply_multi_stage_review_fusion(scored, selector_path, scorer_path, social_hot_features_path=social_path, theme_hot_features_path=theme_path)
    by_code = fused.set_index("ts_code")
    assert by_code.loc["000333.SZ", "social_bonus_score"] == 0.04
    assert by_code.loc["000333.SZ", "theme_bonus_score"] == 0.05
    assert by_code.loc["000333.SZ", "final_live_score"] > by_code.loc["600941.SH", "final_live_score"]
