"""
LLM Pool — Multi-model manager with dynamic scheduling.

Three operating modes (auto-detected):

1. Dynamic scheduling (new):
   - llm_roles uses declarative format: {"mode": "chat", "prefer_cost": "low"}
   - Call schedule_roles(probe_results) after probing
   - Scheduler assigns best model per role based on probe score + cost preference
   - Fallback chains auto-generated from remaining candidates

2. Static assignment (legacy v2):
   - llm_roles uses explicit format: {"model": "gpt54", "mode": "chat", "fallback": [...]}
   - Bypasses scheduler, uses config as-is

3. Legacy fallback (v1):
   - No llm_roles configured
   - Falls back to deep_think_llm / quick_think_llm

Usage:
    pool = LLMPool(config)
    pool.schedule_roles(probe_results)          # dynamic scheduling
    llm = pool.get_llm("market_analyst")        # returns best available model
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

from llm import create_llm_client

logger = logging.getLogger(__name__)

# Default role → tier (used only for legacy v1 fallback)
_ROLE_TIER = {
    "research_manager": "deep",
    "portfolio_manager": "deep",
    "reflector": "deep",
    "market_analyst": "quick",
    "news_analyst": "quick",
    "social_analyst": "quick",
    "fundamentals_analyst": "quick",
    "bull_researcher": "quick",
    "bear_researcher": "quick",
    "trader": "quick",
    "aggressive_debater": "quick",
    "conservative_debater": "quick",
    "neutral_debater": "quick",
    "signal_processor": "quick",
}

_TRANSIENT_KEYWORDS = (
    "timeout", "connection", "reset", "refused", "remote",
    "rate limit", "too many requests", "429", "503", "502",
)

# Cost preference scoring adjustments
_COST_BONUS = {
    # (prefer_cost, actual_cost_tier) → score adjustment
    ("low", "low"): 15,
    ("low", "medium"): 0,
    ("low", "high"): -15,
    ("medium", "low"): 5,
    ("medium", "medium"): 10,
    ("medium", "high"): -5,
    ("high", "low"): -5,
    ("high", "medium"): 0,
    ("high", "high"): 10,
    # "any" → no adjustment (handled in code)
}


class ResilientLLM:
    """LLM wrapper with automatic retry and failover.

    Tries the primary model first. On transient errors, retries up to
    max_retries times, then moves to the next fallback model.
    Non-transient errors skip retries and move to the next model immediately.
    """

    def __init__(self, primary, fallbacks: List = None, max_retries: int = 2):
        self.primary = primary
        self.fallbacks = fallbacks or []
        self.max_retries = max_retries

    def invoke(self, *args, **kwargs):
        chain = [self.primary] + self.fallbacks
        last_error = None
        for i, llm in enumerate(chain):
            for attempt in range(self.max_retries):
                try:
                    return llm.invoke(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if self._is_transient(e):
                        wait = min(2 ** attempt, 8)
                        logger.warning(
                            "Transient error on model %d attempt %d, retry in %ds: %s",
                            i, attempt + 1, wait, e,
                        )
                        time.sleep(wait)
                        continue
                    logger.warning("Non-transient error on model %d: %s", i, e)
                    break
        raise last_error

    @staticmethod
    def _is_transient(e: Exception) -> bool:
        msg = str(e).lower()
        return any(kw in msg for kw in _TRANSIENT_KEYWORDS)

    def __getattr__(self, name):
        return getattr(self.primary, name)


class LLMPool:
    """Multi-model pool with dynamic scheduling, mode support, and failover."""

    def __init__(self, config: Dict[str, Any], callbacks: list = None):
        self.config = config
        self.callbacks = callbacks or []
        self._cache: Dict[str, Any] = {}          # "model_key:mode" → LLM instance
        self._legacy_cache: Dict[str, Any] = {}   # "deep"/"quick" → LLM
        self._disabled_models: Set[str] = set()
        self._no_deepthink: Set[str] = set()
        self._assignments: Dict[str, Dict] = {}   # role → {"model", "mode", "fallbacks"}
        self._scheduled: bool = False              # True after schedule_roles() runs

    # ------------------------------------------------------------------
    # Dynamic scheduling
    # ------------------------------------------------------------------

    def schedule_roles(self, probe_results: Dict) -> Dict[str, Dict]:
        """Assign models to roles based on probe results and role requirements.

        For each role in llm_roles:
          - If declarative format (has prefer_cost, no "model"): schedule dynamically
          - If static format (has "model"): use as-is, skip scheduling
          - If string: treat as static chat assignment

        Args:
            probe_results: Dict of model_key → ProbeResult from LLMProbe

        Returns:
            Dict of role → assignment {"model", "mode", "fallbacks"}
        """
        # Mark unavailable models
        for model_key, result in probe_results.items():
            if not result.available:
                self._disabled_models.add(model_key)
            elif not result.deepthink_ok:
                self._no_deepthink.add(model_key)

        # Build available model list with their probe data
        available = {
            k: r for k, r in probe_results.items() if r.available
        }

        llm_roles = self.config.get("llm_roles", {})

        for role, cfg in llm_roles.items():
            # --- Detect config format ---
            if isinstance(cfg, str):
                # Legacy string: "model-key" → static chat
                self._assignments[role] = {
                    "model": cfg, "mode": "chat", "fallbacks": [],
                }
                continue

            if "model" in cfg:
                # Static v2 format: {"model": "xxx", "mode": "...", "fallback": [...]}
                fallbacks = [
                    (fb["model"], fb.get("mode", "chat"))
                    for fb in cfg.get("fallback", [])
                ]
                self._assignments[role] = {
                    "model": cfg["model"],
                    "mode": cfg.get("mode", "chat"),
                    "fallbacks": fallbacks,
                }
                continue

            # --- Declarative format: schedule dynamically ---
            mode = cfg.get("mode", "chat")
            prefer_cost = cfg.get("prefer_cost", "any")

            candidates = self._rank_candidates(available, mode, prefer_cost)

            if not candidates:
                # No model supports required mode → try downgrade to chat
                if mode == "deepthink":
                    logger.warning(
                        "Role '%s' needs deepthink but no model supports it, "
                        "downgrading to chat", role,
                    )
                    candidates = self._rank_candidates(available, "chat", prefer_cost)

            if not candidates:
                logger.error("Role '%s': no available model at all", role)
                continue

            best = candidates[0]
            fallbacks = candidates[1:]

            self._assignments[role] = {
                "model": best[0],
                "mode": best[1],
                "fallbacks": [(c[0], c[1]) for c in fallbacks],
            }

            logger.info(
                "Scheduled '%s' → %s:%s (score=%.1f) fallbacks=%s",
                role, best[0], best[1], best[2],
                [f"{c[0]}:{c[1]}" for c in fallbacks],
            )

        self._scheduled = True
        return dict(self._assignments)

    def _rank_candidates(
        self, available: Dict, mode: str, prefer_cost: str,
    ) -> List[tuple]:
        """Rank available models for a role requirement.

        Returns list of (model_key, actual_mode, effective_score) sorted desc.
        """
        candidates = []
        pool_cfg = self.config.get("llm_pool", {})

        for model_key, result in available.items():
            actual_mode = mode

            # Deepthink requested but model can't do it → skip
            if mode == "deepthink" and not result.deepthink_ok:
                continue

            # Get cost_tier from pool config (authoritative) or probe (fallback)
            model_cfg = pool_cfg.get(model_key, {})
            cost_tier = model_cfg.get("cost_tier", result.cost_tier)

            # Compute effective score
            effective = self._compute_effective_score(
                result.score, cost_tier, prefer_cost,
            )

            candidates.append((model_key, actual_mode, effective))

        # Sort by effective score descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates

    @staticmethod
    def _compute_effective_score(
        base_score: float, cost_tier: str, prefer_cost: str,
    ) -> float:
        """Compute effective score = probe score + cost preference bonus."""
        if prefer_cost == "any":
            return base_score

        bonus = _COST_BONUS.get((prefer_cost, cost_tier), 0)
        return base_score + bonus

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_llm(self, role: str) -> Any:
        """Get LLM instance for a specific agent role.

        Resolution order:
          1. _assignments (from schedule_roles)
          2. llm_roles config (static/legacy format)
          3. Legacy deep_think_llm / quick_think_llm
        """
        # 1. Dynamic assignment (after scheduling)
        if role in self._assignments:
            return self._build_from_assignment(self._assignments[role])

        # 2. Static config fallback (if schedule_roles wasn't called)
        llm_roles = self.config.get("llm_roles", {})
        role_cfg = llm_roles.get(role)

        if role_cfg is not None:
            if isinstance(role_cfg, str):
                return self.get_llm_by_key(role_cfg, mode="chat")
            if "model" in role_cfg:
                return self._build_static_role(role_cfg)
            # Declarative format but schedule_roles not called → pick first available
            if not self._scheduled:
                logger.warning(
                    "Role '%s' uses declarative config but schedule_roles() "
                    "not called. Falling back to first available model.", role,
                )
                return self._fallback_first_available(role_cfg)

        # 3. Legacy fallback
        tier = _ROLE_TIER.get(role, "quick")
        return self._get_legacy_llm(tier)

    def get_llm_by_key(self, model_key: str, mode: str = "chat") -> Any:
        """Get LLM instance by model pool key and mode."""
        cache_key = f"{model_key}:{mode}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        pool = self.config.get("llm_pool", {})
        if model_key not in pool:
            raise ValueError(
                f"Model '{model_key}' not found in llm_pool. "
                f"Available: {list(pool.keys())}"
            )

        if model_key in self._disabled_models:
            raise RuntimeError(f"Model '{model_key}' is unavailable (probe failed)")

        actual_mode = mode
        if mode == "deepthink" and model_key in self._no_deepthink:
            logger.warning(
                "Model '%s' has no deepthink capability, downgrading to chat",
                model_key,
            )
            actual_mode = "chat"
            cache_key = f"{model_key}:{actual_mode}"
            if cache_key in self._cache:
                return self._cache[cache_key]

        model_cfg = pool[model_key]
        llm = self._create_llm(model_cfg, mode=actual_mode)
        self._cache[cache_key] = llm

        logger.info(
            "LLMPool: initialized '%s' mode=%s (%s @ %s)",
            model_key, actual_mode, model_cfg["model"],
            model_cfg.get("base_url", "default"),
        )
        return llm

    def get_all_keys(self) -> list:
        return list(self.config.get("llm_pool", {}).keys())

    def get_role_mapping(self) -> Dict[str, Any]:
        """Return current role assignments (dynamic if scheduled, else config)."""
        if self._assignments:
            return dict(self._assignments)
        return dict(self.config.get("llm_roles", {}))

    def get_schedule_summary(self) -> str:
        """Human-readable summary of current assignments."""
        if not self._assignments:
            return "No scheduling performed yet."
        lines = []
        for role, a in self._assignments.items():
            fb = ", ".join(f"{f[0]}:{f[1]}" for f in a.get("fallbacks", []))
            fb_str = f" fallback=[{fb}]" if fb else ""
            lines.append(f"  {role:<25} → {a['model']}:{a['mode']}{fb_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal: build LLM from assignment
    # ------------------------------------------------------------------

    def _build_from_assignment(self, assignment: Dict) -> Any:
        """Build LLM (with ResilientLLM if fallbacks exist) from a schedule assignment."""
        model_key = assignment["model"]
        mode = assignment["mode"]
        fallback_specs = assignment.get("fallbacks", [])

        # Build primary
        primary = None
        if model_key not in self._disabled_models:
            try:
                primary = self.get_llm_by_key(model_key, mode=mode)
            except Exception as e:
                logger.warning("Failed to create primary '%s:%s': %s", model_key, mode, e)

        # Build fallbacks
        fallbacks = []
        for fb_key, fb_mode in fallback_specs:
            if fb_key in self._disabled_models:
                continue
            try:
                fb_llm = self.get_llm_by_key(fb_key, mode=fb_mode)
                fallbacks.append(fb_llm)
            except Exception as e:
                logger.warning("Failed to create fallback '%s:%s': %s", fb_key, fb_mode, e)

        if primary is None and not fallbacks:
            raise RuntimeError(
                f"No available model for assignment: {assignment}"
            )

        if primary is None:
            primary = fallbacks.pop(0)

        if not fallbacks:
            return primary

        return ResilientLLM(primary, fallbacks)

    def _build_static_role(self, role_cfg: Dict) -> Any:
        """Build LLM from static v2 format: {"model": "xxx", "mode": "...", "fallback": [...]}."""
        model_key = role_cfg["model"]
        mode = role_cfg.get("mode", "chat")
        fallback_cfgs = role_cfg.get("fallback", [])

        primary = None
        if model_key not in self._disabled_models:
            try:
                primary = self.get_llm_by_key(model_key, mode=mode)
            except Exception as e:
                logger.warning("Failed to create primary '%s:%s': %s", model_key, mode, e)

        fallbacks = []
        for fb_cfg in fallback_cfgs:
            fb_key = fb_cfg["model"]
            fb_mode = fb_cfg.get("mode", "chat")
            if fb_key in self._disabled_models:
                continue
            try:
                fallbacks.append(self.get_llm_by_key(fb_key, mode=fb_mode))
            except Exception as e:
                logger.warning("Failed to create fallback '%s:%s': %s", fb_key, fb_mode, e)

        if primary is None and not fallbacks:
            raise RuntimeError(f"No available model for role config {role_cfg}")

        if primary is None:
            primary = fallbacks.pop(0)

        if not fallbacks:
            return primary

        return ResilientLLM(primary, fallbacks)

    def _fallback_first_available(self, role_cfg: Dict) -> Any:
        """Fallback when scheduler wasn't run: pick best model using cost_tier
        from pool config (no probe needed).

        Instead of blindly picking the first model in the dict, we rank
        candidates by cost preference bonus — the same logic schedule_roles
        uses, but with a fixed base score so the ranking is purely
        cost-driven.
        """
        mode = role_cfg.get("mode", "chat")
        prefer_cost = role_cfg.get("prefer_cost", "any")
        pool = self.config.get("llm_pool", {})

        # Build (model_key, effective_score) pairs
        candidates: list[tuple[str, float]] = []
        for model_key, model_cfg in pool.items():
            if model_key in self._disabled_models:
                continue
            if mode == "deepthink" and model_key in self._no_deepthink:
                continue
            cost_tier = model_cfg.get("cost_tier", "medium")
            score = self._compute_effective_score(50.0, cost_tier, prefer_cost)
            candidates.append((model_key, score))

        # Sort by score descending (best match first)
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Try candidates in ranked order; build primary + fallbacks
        primary = None
        fallbacks = []
        for model_key, _score in candidates:
            try:
                llm = self.get_llm_by_key(model_key, mode=mode)
                if primary is None:
                    primary = llm
                else:
                    fallbacks.append(llm)
            except Exception:
                continue

        if primary is not None:
            if fallbacks:
                logger.info(
                    "Fallback assignment: mode=%s prefer_cost=%s → %s "
                    "(+%d fallbacks)",
                    mode, prefer_cost, candidates[0][0], len(fallbacks),
                )
                return ResilientLLM(primary, fallbacks)
            logger.info(
                "Fallback assignment: mode=%s prefer_cost=%s → %s",
                mode, prefer_cost, candidates[0][0],
            )
            return primary

        # Deepthink failed → downgrade to chat and retry
        if mode == "deepthink":
            logger.warning(
                "No model available for deepthink, downgrading to chat",
            )
            role_cfg_chat = dict(role_cfg, mode="chat")
            return self._fallback_first_available(role_cfg_chat)

        raise RuntimeError("No available model in pool")

    # ------------------------------------------------------------------
    # Internal: create LLM instance
    # ------------------------------------------------------------------

    def _create_llm(self, model_cfg: Dict[str, Any], mode: str = "chat") -> Any:
        """Create an LLM client from a pool entry config with mode-specific kwargs."""
        provider = model_cfg.get("provider", "openai")
        model = model_cfg["model"]
        base_url = model_cfg.get("base_url")

        api_key = self._resolve_api_key(model_cfg)

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if self.callbacks:
            kwargs["callbacks"] = self.callbacks

        for key in ("timeout", "max_tokens"):
            if key in model_cfg:
                kwargs[key] = model_cfg[key]

        # Merge mode-specific kwargs
        modes = model_cfg.get("modes", {})
        mode_kwargs = modes.get(mode, {})
        kwargs.update(mode_kwargs)

        client = create_llm_client(
            provider=provider, model=model, base_url=base_url, **kwargs,
        )
        return client.get_llm()

    @staticmethod
    def _resolve_api_key(model_cfg: Dict) -> Optional[str]:
        api_key = None
        api_key_env = model_cfg.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(api_key_env)
        if not api_key:
            api_key = model_cfg.get("api_key")
        return api_key

    def _get_legacy_llm(self, tier: str) -> Any:
        """Fallback: create LLM from legacy deep_think_llm/quick_think_llm config."""
        if tier in self._legacy_cache:
            return self._legacy_cache[tier]

        model = self.config.get(
            "deep_think_llm" if tier == "deep" else "quick_think_llm"
        )

        kwargs = {}
        if self.callbacks:
            kwargs["callbacks"] = self.callbacks

        provider = self.config.get("llm_provider", "openai")
        for cfg_key, kwarg_key in [
            ("google_thinking_level", "thinking_level"),
            ("openai_reasoning_effort", "reasoning_effort"),
            ("anthropic_effort", "effort"),
        ]:
            val = self.config.get(cfg_key)
            if val:
                kwargs[kwarg_key] = val

        client = create_llm_client(
            provider=provider, model=model,
            base_url=self.config.get("backend_url"), **kwargs,
        )
        llm = client.get_llm()
        self._legacy_cache[tier] = llm
        return llm
