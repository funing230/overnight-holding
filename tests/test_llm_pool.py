"""
Unit tests for LLMPool — dynamic scheduling, mode support, and failover.
"""

import pytest
from unittest.mock import patch, MagicMock
from dataclasses import dataclass, field
from typing import List

from llm.pool import LLMPool, ResilientLLM


# ===================================================================
# Fake probe result
# ===================================================================

@dataclass
class FakeProbeResult:
    available: bool = True
    deepthink_ok: bool = True
    score: float = 80.0
    cost_tier: str = "medium"


# ===================================================================
# Test configs
# ===================================================================

# New declarative format
DECLARATIVE_CONFIG = {
    "llm_pool": {
        "model-cheap": {
            "provider": "openai",
            "model": "test-cheap",
            "base_url": "http://localhost:1111/v1",
            "api_key": "key-a",
            "cost_tier": "low",
            "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        },
        "model-mid": {
            "provider": "openai",
            "model": "test-mid",
            "base_url": "http://localhost:2222/v1",
            "api_key": "key-b",
            "cost_tier": "medium",
            "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        },
        "model-expensive": {
            "provider": "openai",
            "model": "test-expensive",
            "base_url": "http://localhost:3333/v1",
            "api_key": "key-c",
            "cost_tier": "high",
            "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        },
    },
    "llm_roles": {
        "market_analyst":    {"mode": "chat",      "prefer_cost": "low"},
        "research_manager":  {"mode": "deepthink", "prefer_cost": "any"},
        "trader":            {"mode": "deepthink", "prefer_cost": "any"},
        "signal_processor":  {"mode": "chat",      "prefer_cost": "low"},
    },
    "llm_provider": "openai",
    "deep_think_llm": "legacy-deep",
    "quick_think_llm": "legacy-quick",
    "backend_url": "http://localhost:9999/v1",
}

# Static v2 format (backward compat)
STATIC_V2_CONFIG = {
    "llm_pool": {
        "model-a": {
            "provider": "openai",
            "model": "test-a",
            "base_url": "http://localhost:1111/v1",
            "api_key": "key-a",
        },
        "model-b": {
            "provider": "openai",
            "model": "test-b",
            "base_url": "http://localhost:2222/v1",
            "api_key": "key-b",
        },
    },
    "llm_roles": {
        "market_analyst":   {"model": "model-a", "mode": "chat"},
        "research_manager": {"model": "model-b", "mode": "deepthink",
                             "fallback": [{"model": "model-a", "mode": "deepthink"}]},
    },
    "llm_provider": "openai",
    "deep_think_llm": "legacy-deep",
    "quick_think_llm": "legacy-quick",
    "backend_url": "http://localhost:9999/v1",
}

# Legacy string format
LEGACY_STRING_CONFIG = {
    "llm_pool": {
        "model-a": {
            "provider": "openai",
            "model": "test-a",
            "base_url": "http://localhost:1111/v1",
            "api_key": "key-a",
        },
    },
    "llm_roles": {
        "market_analyst": "model-a",
    },
    "llm_provider": "openai",
    "deep_think_llm": "legacy-deep",
    "quick_think_llm": "legacy-quick",
    "backend_url": "http://localhost:9999/v1",
}

# No llm_roles at all
LEGACY_V1_CONFIG = {
    "llm_provider": "openai",
    "deep_think_llm": "legacy-deep",
    "quick_think_llm": "legacy-quick",
    "backend_url": "http://localhost:9999/v1",
}


# ===================================================================
# Tests — Dynamic scheduling
# ===================================================================


class TestScheduleRoles:
    """Test schedule_roles() with declarative config."""

    def test_low_cost_role_prefers_cheap_model(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, cost_tier="high"),
        }
        assignments = pool.schedule_roles(probe)

        # market_analyst wants chat + low cost
        # model-cheap: 80 + 15 = 95
        # model-mid:   85 + 0  = 85
        # model-expensive: 90 - 15 = 75
        assert assignments["market_analyst"]["model"] == "model-cheap"

    def test_any_cost_role_prefers_highest_score(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, cost_tier="high"),
        }
        assignments = pool.schedule_roles(probe)

        # research_manager wants deepthink + any cost → pure score ranking
        assert assignments["research_manager"]["model"] == "model-expensive"

    def test_fallbacks_auto_generated(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, cost_tier="high"),
        }
        assignments = pool.schedule_roles(probe)

        # market_analyst: best=model-cheap, fallbacks=remaining sorted by effective score
        fb_models = [f[0] for f in assignments["market_analyst"]["fallbacks"]]
        assert len(fb_models) == 2
        assert "model-mid" in fb_models
        assert "model-expensive" in fb_models

    def test_unavailable_model_excluded(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(available=False, score=0),
            "model-mid":       FakeProbeResult(score=85, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, cost_tier="high"),
        }
        assignments = pool.schedule_roles(probe)

        # model-cheap is down → not assigned to any role
        assert assignments["market_analyst"]["model"] != "model-cheap"
        for fb in assignments["market_analyst"]["fallbacks"]:
            assert fb[0] != "model-cheap"

    def test_deepthink_role_skips_no_deepthink_model(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, deepthink_ok=False, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, deepthink_ok=True, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, deepthink_ok=True, cost_tier="high"),
        }
        assignments = pool.schedule_roles(probe)

        # research_manager needs deepthink → model-cheap excluded
        assert assignments["research_manager"]["model"] != "model-cheap"
        for fb in assignments["research_manager"]["fallbacks"]:
            assert fb[0] != "model-cheap"

    def test_deepthink_downgrades_to_chat_when_no_model_supports(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, deepthink_ok=False, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, deepthink_ok=False, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, deepthink_ok=False, cost_tier="high"),
        }
        assignments = pool.schedule_roles(probe)

        # No model has deepthink → research_manager should downgrade to chat
        assert assignments["research_manager"]["model"] is not None
        # Still gets assigned (downgraded)

    def test_all_models_down_skips_role(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(available=False),
            "model-mid":       FakeProbeResult(available=False),
            "model-expensive": FakeProbeResult(available=False),
        }
        assignments = pool.schedule_roles(probe)

        # No model available → role not in assignments
        assert "market_analyst" not in assignments

    def test_mode_preserved_in_assignment(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, cost_tier="high"),
        }
        assignments = pool.schedule_roles(probe)

        assert assignments["market_analyst"]["mode"] == "chat"
        assert assignments["research_manager"]["mode"] == "deepthink"


class TestScheduleWithGetLLM:
    """Test that get_llm uses scheduled assignments."""

    @patch("llm.pool.create_llm_client")
    def test_get_llm_uses_scheduled_model(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, cost_tier="high"),
        }
        pool.schedule_roles(probe)

        pool.get_llm("market_analyst")

        # Should create model-cheap (best for low cost chat)
        call_kwargs = mock_create.call_args_list[0][1]
        assert call_kwargs["model"] == "test-cheap"

    @patch("llm.pool.create_llm_client")
    def test_get_llm_returns_resilient_when_fallbacks(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, cost_tier="high"),
        }
        pool.schedule_roles(probe)

        llm = pool.get_llm("market_analyst")
        # Has fallbacks → should be ResilientLLM
        assert isinstance(llm, ResilientLLM)

    @patch("llm.pool.create_llm_client")
    def test_deepthink_role_gets_reasoning_kwargs(self, mock_create):
        mock_client = MagicMock()
        mock_client.get_llm.return_value = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, cost_tier="high"),
        }
        pool.schedule_roles(probe)

        pool.get_llm("research_manager")

        # research_manager → model-expensive:deepthink → should have reasoning_effort
        first_call = mock_create.call_args_list[0][1]
        assert first_call["model"] == "test-expensive"
        assert first_call["reasoning_effort"] == "high"


# ===================================================================
# Tests — Backward compatibility
# ===================================================================


class TestStaticV2Compat:
    """Test static v2 format still works after scheduling."""

    def test_static_format_bypasses_scheduler(self):
        pool = LLMPool(STATIC_V2_CONFIG)
        probe = {
            "model-a": FakeProbeResult(score=80),
            "model-b": FakeProbeResult(score=90),
        }
        assignments = pool.schedule_roles(probe)

        # Static format: model is preserved as-is
        assert assignments["market_analyst"]["model"] == "model-a"
        assert assignments["research_manager"]["model"] == "model-b"

    def test_static_format_preserves_fallback(self):
        pool = LLMPool(STATIC_V2_CONFIG)
        probe = {
            "model-a": FakeProbeResult(score=80),
            "model-b": FakeProbeResult(score=90),
        }
        assignments = pool.schedule_roles(probe)

        fb = assignments["research_manager"]["fallbacks"]
        assert ("model-a", "deepthink") in fb


class TestLegacyStringCompat:
    """Test legacy string format."""

    def test_string_format_treated_as_chat(self):
        pool = LLMPool(LEGACY_STRING_CONFIG)
        probe = {
            "model-a": FakeProbeResult(score=80),
        }
        assignments = pool.schedule_roles(probe)

        assert assignments["market_analyst"]["model"] == "model-a"
        assert assignments["market_analyst"]["mode"] == "chat"


class TestLegacyV1Compat:
    """Test legacy v1 (no llm_roles) fallback."""

    @patch("llm.pool.create_llm_client")
    def test_no_roles_uses_legacy_llm(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(LEGACY_V1_CONFIG)
        pool.get_llm("market_analyst")

        assert mock_create.call_args[1]["model"] == "legacy-quick"

    @patch("llm.pool.create_llm_client")
    def test_decision_role_uses_deep(self, mock_create):
        mock_client = MagicMock()
        mock_create.return_value = mock_client

        pool = LLMPool(LEGACY_V1_CONFIG)
        pool.get_llm("research_manager")

        assert mock_create.call_args[1]["model"] == "legacy-deep"


# ===================================================================
# Tests — ResilientLLM
# ===================================================================


class TestResilientLLM:
    """Test retry and failover logic."""

    def test_primary_success(self):
        primary = MagicMock()
        primary.invoke.return_value = "ok"
        fallback = MagicMock()

        llm = ResilientLLM(primary, [fallback])
        assert llm.invoke("test") == "ok"
        fallback.invoke.assert_not_called()

    def test_transient_error_retries(self):
        primary = MagicMock()
        primary.invoke.side_effect = [ConnectionError("timeout"), "ok"]

        llm = ResilientLLM(primary, [], max_retries=2)
        assert llm.invoke("test") == "ok"
        assert primary.invoke.call_count == 2

    def test_primary_fails_uses_fallback(self):
        primary = MagicMock()
        primary.invoke.side_effect = ConnectionError("timeout")
        fallback = MagicMock()
        fallback.invoke.return_value = "fallback-ok"

        llm = ResilientLLM(primary, [fallback], max_retries=1)
        assert llm.invoke("test") == "fallback-ok"

    def test_non_transient_skips_retries(self):
        primary = MagicMock()
        primary.invoke.side_effect = ValueError("bad input")
        fallback = MagicMock()
        fallback.invoke.return_value = "fallback-ok"

        llm = ResilientLLM(primary, [fallback], max_retries=3)
        assert llm.invoke("test") == "fallback-ok"
        assert primary.invoke.call_count == 1

    def test_all_fail_raises(self):
        primary = MagicMock()
        primary.invoke.side_effect = ConnectionError("timeout")
        fallback = MagicMock()
        fallback.invoke.side_effect = ConnectionError("timeout")

        llm = ResilientLLM(primary, [fallback], max_retries=1)
        with pytest.raises(ConnectionError):
            llm.invoke("test")


# ===================================================================
# Tests — Scoring
# ===================================================================


class TestScoring:
    """Test _compute_effective_score and _rank_candidates."""

    def test_low_cost_preference_boosts_cheap(self):
        score = LLMPool._compute_effective_score(80, "low", "low")
        assert score == 95  # 80 + 15

    def test_low_cost_preference_penalizes_expensive(self):
        score = LLMPool._compute_effective_score(90, "high", "low")
        assert score == 75  # 90 - 15

    def test_any_cost_no_adjustment(self):
        score = LLMPool._compute_effective_score(85, "high", "any")
        assert score == 85  # no change

    def test_medium_preference_favors_medium(self):
        score = LLMPool._compute_effective_score(80, "medium", "medium")
        assert score == 90  # 80 + 10


# ===================================================================
# Tests — Metadata
# ===================================================================


class TestMetadata:
    def test_get_all_keys(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        assert set(pool.get_all_keys()) == {"model-cheap", "model-mid", "model-expensive"}

    def test_get_schedule_summary_before_scheduling(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        assert "No scheduling" in pool.get_schedule_summary()

    def test_get_schedule_summary_after_scheduling(self):
        pool = LLMPool(DECLARATIVE_CONFIG)
        probe = {
            "model-cheap":     FakeProbeResult(score=80, cost_tier="low"),
            "model-mid":       FakeProbeResult(score=85, cost_tier="medium"),
            "model-expensive": FakeProbeResult(score=90, cost_tier="high"),
        }
        pool.schedule_roles(probe)
        summary = pool.get_schedule_summary()
        assert "market_analyst" in summary
        assert "model-cheap" in summary
