"""
LLM Probe — Dynamic model capability discovery.

Tests each model in the pool for:
  1. Connectivity (is it reachable?)
  2. Chat capability (can it answer simple questions?)
  3. DeepThink capability (does it produce reasoning chains?)
  4. Latency measurement
  5. DeepThink quality (structured vs textual)
  6. Composite scoring for scheduler ranking

Usage:
    probe = LLMProbe(config)
    results = probe.probe_all()
    result = probe.probe_model("claude-opus")
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Probe prompts
# ---------------------------------------------------------------------------

CHAT_PROBE_PROMPT = "What is 2+3? Reply with just the number."

DEEPTHINK_PROBE_PROMPT = (
    "A farmer has 17 sheep. All but 9 run away. "
    "How many sheep does he have left? Think step by step."
)

_REASONING_MARKERS = [
    "step", "because", "therefore", "thus", "so ",
    "first", "second", "let me", "think",
    "分析", "因为", "所以", "首先", "其次", "推理",
]

PROBE_TIMEOUT = 60

# Cost tier defaults (can be overridden in pool config)
_DEFAULT_COST_TIERS = {
    "claude": "high",
    "gpt": "medium",
    "gemini": "low",
    "deepseek": "low",
    "qwen": "low",
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    """Result of probing a single model."""
    model_key: str
    available: bool = False
    chat_ok: bool = False
    deepthink_ok: bool = False
    capabilities: List[str] = field(default_factory=list)
    latency_chat_ms: float = 0.0
    latency_deepthink_ms: float = 0.0
    chat_response: str = ""
    deepthink_response: str = ""
    error: str = ""
    context_window: int = 0
    model_name: str = ""
    base_url: str = ""

    # New fields
    cost_tier: str = "unknown"              # high / medium / low
    deepthink_quality: str = "none"         # structured / textual / none
    score: float = 0.0                      # composite score 0-100
    chat_correct: bool = False              # did chat answer 5?
    deepthink_correct: bool = False         # did deepthink answer 9?

    @property
    def best_mode(self) -> str:
        if self.deepthink_ok:
            return "deepthink"
        if self.chat_ok:
            return "chat"
        return "unavailable"

    def compute_score(self) -> float:
        """Compute composite score (0-100) for scheduler ranking.

        Scoring breakdown:
          - Available:              30 pts
          - Chat OK:                15 pts
          - Chat correct answer:     5 pts
          - DeepThink OK:           15 pts
          - DeepThink correct:       5 pts
          - Structured reasoning:   10 pts (vs textual 5 pts)
          - Speed bonus:         0-15 pts (faster = higher)
          - Context bonus:        0-5 pts (larger = higher)
        """
        if not self.available:
            self.score = 0.0
            return self.score

        s = 30.0  # base: available

        # Chat
        if self.chat_ok:
            s += 15.0
        if self.chat_correct:
            s += 5.0

        # DeepThink
        if self.deepthink_ok:
            s += 15.0
        if self.deepthink_correct:
            s += 5.0

        # Reasoning quality
        if self.deepthink_quality == "structured":
            s += 10.0
        elif self.deepthink_quality == "textual":
            s += 5.0

        # Speed bonus: 15 pts max, 0 pts at 15s+
        if self.latency_chat_ms > 0:
            speed_score = max(0.0, 15.0 - (self.latency_chat_ms / 1000.0))
            s += speed_score

        # Context window bonus: 0-5 pts
        if self.context_window > 0:
            # 200k = 5pts, 100k = 2.5pts, 2M = 5pts (capped)
            ctx_score = min(5.0, self.context_window / 40000.0)
            s += ctx_score

        self.score = min(100.0, round(s, 1))
        return self.score

    def summary(self) -> str:
        status = "✅" if self.available else "❌"
        caps = "+".join(self.capabilities) if self.capabilities else "none"
        latency = f"{self.latency_chat_ms:.0f}ms"
        quality = f" quality={self.deepthink_quality}" if self.deepthink_ok else ""
        return (
            f"{status} {self.model_key}: {caps}{quality} | "
            f"{latency} | ctx={self.context_window} | "
            f"cost={self.cost_tier} | score={self.score}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for logging/storage."""
        return {
            "model_key": self.model_key,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "available": self.available,
            "capabilities": self.capabilities,
            "deepthink_quality": self.deepthink_quality,
            "cost_tier": self.cost_tier,
            "score": self.score,
            "latency_chat_ms": round(self.latency_chat_ms, 0),
            "latency_deepthink_ms": round(self.latency_deepthink_ms, 0),
            "context_window": self.context_window,
            "chat_correct": self.chat_correct,
            "deepthink_correct": self.deepthink_correct,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Probe implementation
# ---------------------------------------------------------------------------


class LLMProbe:
    """Probe models in the pool for availability and capabilities."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool = config.get("llm_pool", {})

    def probe_all(self, verbose: bool = True) -> Dict[str, ProbeResult]:
        """Probe all models in the pool."""
        results = {}
        for model_key in self.pool:
            if verbose:
                print(f"[Probe] Testing {model_key}...")
            result = self.probe_model(model_key)
            results[model_key] = result
            if verbose:
                print(f"  {result.summary()}")
        return results

    def probe_model(self, model_key: str) -> ProbeResult:
        """Probe a single model for connectivity and capabilities."""
        if model_key not in self.pool:
            return ProbeResult(
                model_key=model_key,
                error=f"Model '{model_key}' not in pool",
            )

        model_cfg = self.pool[model_key]
        result = ProbeResult(
            model_key=model_key,
            model_name=model_cfg.get("model", ""),
            base_url=model_cfg.get("base_url", ""),
            context_window=model_cfg.get("context_window", 0),
        )

        # Determine cost tier
        result.cost_tier = self._determine_cost_tier(model_key, model_cfg)

        # Build LLM instance
        try:
            llm = self._create_llm(model_cfg)
        except Exception as e:
            result.error = f"Failed to create LLM: {e}"
            result.compute_score()
            return result

        # Phase 1: Chat probe
        try:
            t0 = time.time()
            chat_resp = llm.invoke(CHAT_PROBE_PROMPT)
            t1 = time.time()

            content = self._extract_content(chat_resp)
            result.latency_chat_ms = (t1 - t0) * 1000
            result.chat_response = content
            result.available = True
            result.chat_ok = True
            result.capabilities.append("chat")
            result.chat_correct = "5" in content
        except Exception as e:
            result.error = f"Chat probe failed: {e}"
            result.compute_score()
            return result

        # Phase 2: DeepThink probe
        try:
            t0 = time.time()
            think_resp = llm.invoke(DEEPTHINK_PROBE_PROMPT)
            t1 = time.time()

            content = self._extract_content(think_resp)
            result.latency_deepthink_ms = (t1 - t0) * 1000
            result.deepthink_response = content
            result.deepthink_correct = self._check_answer_is_9(content)

            # Detect reasoning type
            reasoning_type = self._classify_reasoning(content, think_resp)
            result.deepthink_quality = reasoning_type

            if reasoning_type != "none" and result.deepthink_correct:
                result.deepthink_ok = True
                result.capabilities.append("deepthink")
        except Exception as e:
            result.error = f"DeepThink probe failed: {e}"

        result.compute_score()
        return result

    def _determine_cost_tier(self, model_key: str, model_cfg: Dict) -> str:
        """Determine cost tier from config or model name heuristic."""
        # Explicit config takes priority
        if "cost_tier" in model_cfg:
            return model_cfg["cost_tier"]

        # Heuristic from model key / model name
        key_lower = model_key.lower()
        name_lower = model_cfg.get("model", "").lower()
        combined = key_lower + " " + name_lower

        for keyword, tier in _DEFAULT_COST_TIERS.items():
            if keyword in combined:
                return tier

        return "medium"

    def _create_llm(self, model_cfg: Dict[str, Any]):
        """Create an LLM instance from pool config."""
        from llm import create_llm_client

        provider = model_cfg.get("provider", "openai")
        model = model_cfg["model"]
        base_url = model_cfg.get("base_url")

        api_key = None
        api_key_env = model_cfg.get("api_key_env")
        if api_key_env:
            api_key = os.environ.get(api_key_env)
        if not api_key:
            api_key = model_cfg.get("api_key")

        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        kwargs["timeout"] = PROBE_TIMEOUT

        client = create_llm_client(
            provider=provider,
            model=model,
            base_url=base_url,
            **kwargs,
        )
        return client.get_llm()

    def _extract_content(self, response) -> str:
        """Extract text content from LLM response."""
        if hasattr(response, "content"):
            content = response.content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if isinstance(block, dict):
                        parts.append(block.get("text", str(block)))
                    elif isinstance(block, str):
                        parts.append(block)
                    else:
                        parts.append(str(block))
                return " ".join(parts)
            return str(content)
        return str(response)

    def _classify_reasoning(self, content: str, raw_response) -> str:
        """Classify reasoning quality: structured / textual / none.

        - structured: explicit thinking tags, reasoning_tokens, or
                      dedicated reasoning blocks (highest quality)
        - textual:    step-by-step reasoning in plain text
        - none:       no detectable reasoning
        """
        content_lower = content.lower()

        # === Structured reasoning detection ===

        # Claude-style <thinking> tags
        if "<thinking>" in content_lower:
            return "structured"

        # API-level reasoning tokens
        if hasattr(raw_response, "additional_kwargs"):
            kwargs = raw_response.additional_kwargs
            if kwargs.get("reasoning_content"):
                return "structured"

        if hasattr(raw_response, "response_metadata"):
            meta = raw_response.response_metadata
            usage = meta.get("usage", meta.get("token_usage", {}))
            if isinstance(usage, dict):
                if usage.get("reasoning_tokens", 0) > 0:
                    return "structured"
                details = usage.get("completion_tokens_details", {})
                if isinstance(details, dict) and details.get("reasoning_tokens", 0) > 0:
                    return "structured"

        # === Textual reasoning detection ===

        marker_count = sum(
            1 for marker in _REASONING_MARKERS
            if marker in content_lower
        )

        if marker_count >= 2 and len(content) > 80:
            return "textual"

        return "none"

    def _check_answer_is_9(self, content: str) -> bool:
        """Check if the response contains the correct answer (9)."""
        return "9" in content


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def probe_models(config: Dict[str, Any], verbose: bool = True) -> Dict[str, ProbeResult]:
    """Probe all models in config and return results."""
    probe = LLMProbe(config)
    return probe.probe_all(verbose=verbose)
