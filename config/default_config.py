import json
import os
from pathlib import Path


def _load_openclaw_provider_map():
    """Load provider credentials from ~/.openclaw/openclaw.json if available."""
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    if not config_path.exists():
        return {}

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    return data.get("models", {}).get("providers", {}) or {}


def _bridge_openclaw_credentials(config):
    """Backfill TradingAgents2.0 llm_pool credentials from OpenClaw config.

    This is a compatibility bridge only. It preserves existing project config,
    but when env vars are missing it reuses the user's OpenClaw provider config
    so TradingAgents2.0 can bootstrap without separate manual exports.
    """
    provider_map = _load_openclaw_provider_map()
    if not provider_map:
        return config

    bridge_map = {
        "gemini": ("custom-api-gemini", "GEMINI_API_KEY"),
        "claude": ("custom-api-claude", "CLAUDE_API_KEY"),
        "claude-opus": ("custom-api-claude", "CLAUDE_API_KEY"),
        "gpt": ("custom-api-gpt54L", "GPT54_API_KEY"),
        "gpt54": ("custom-api-gpt54L", "GPT54_API_KEY"),
    }

    llm_pool = config.get("llm_pool", {})
    for model_key, (provider_key, env_key) in bridge_map.items():
        model_cfg = llm_pool.get(model_key)
        provider_cfg = provider_map.get(provider_key)
        if not model_cfg or not provider_cfg:
            continue

        api_key = provider_cfg.get("apiKey")
        base_url = provider_cfg.get("baseUrl")

        if api_key and not os.environ.get(env_key):
            os.environ.setdefault(env_key, api_key)
        if api_key and not model_cfg.get("api_key"):
            model_cfg["api_key"] = api_key
        if base_url:
            model_cfg["base_url"] = base_url

    return config


DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),

    # =========================================================
    # LLM Pool — register all available models here
    # Each entry: provider, model name, base_url, api_key source
    # "modes" defines per-mode kwargs merged on top of base config
    # Add new models without changing any code
    #
    # Primary production aliases for this project:
    #   - gpt
    #   - claude
    #   - gemini
    # =========================================================
    "llm_pool": {
        "gemini": {
            "provider": "openai",
            "model": "[L]gemini-3-pro-preview",
            "base_url": "https://new.lemonapi.site/v1",
            "api_key_env": "GEMINI_API_KEY",
            "context_window": 2000000,
            "max_tokens": 32768,
            "cost_tier": "low",
            "modes": {
                "chat": {},
                "deepthink": {"thinking_level": "high"},
            },
        },
        "claude": {
            "provider": "openai",
            "model": "claude-opus-4-6-thinking",
            "base_url": "https://www.fucheers.top/v1",
            "api_key_env": "CLAUDE_API_KEY",
            "context_window": 200000,
            "max_tokens": 16384,
            "cost_tier": "high",
            "modes": {
                "chat": {},
                "deepthink": {"reasoning_effort": "high"},
            },
        },
        "gpt": {
            "provider": "openai",
            "model": "gpt-5.4",
            "base_url": "http://92scw.cn/v1",
            "api_key_env": "GPT54_API_KEY",
            "context_window": 128000,
            "max_tokens": 8192,
            "cost_tier": "medium",
            "modes": {
                "chat": {},
                "deepthink": {"reasoning_effort": "high"},
            },
        },
        # === Legacy aliases kept for backward compatibility ===
        "claude-opus": {
            "provider": "openai",
            "model": "claude-opus-4-6-thinking",
            "base_url": "https://www.fucheers.top/v1",
            "api_key_env": "CLAUDE_API_KEY",
            "context_window": 200000,
            "max_tokens": 16384,
            "cost_tier": "high",
            "modes": {
                "chat": {},
                "deepthink": {"reasoning_effort": "high"},
            },
        },
        "gpt54": {
            "provider": "openai",
            "model": "gpt-5.4",
            "base_url": "http://92scw.cn/v1",
            "api_key_env": "GPT54_API_KEY",
            "context_window": 128000,
            "max_tokens": 8192,
            "cost_tier": "medium",
            "modes": {
                "chat": {},
                "deepthink": {"reasoning_effort": "high"},
            },
        },
        # === 扩展模型 ===
        "deepseek": {
            "provider": "openai",
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com/v1",
            "api_key_env": "DEEPSEEK_API_KEY",
            "context_window": 64000,
            "max_tokens": 8192,
            "cost_tier": "low",
            "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        },
        "yuanlan4": {
            "provider": "openai",
            "model": "deepseek-v4-pro",
            "base_url": "https://yuanlansj.xin/v1",
            "api_key_env": "YUANLAN_API_KEY",
            "context_window": 1000000,
            "max_tokens": 8192,
            "cost_tier": "low",
            "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        },
        "qwen37": {
            "provider": "openai",
            "model": "qwen3.7-max",
            "base_url": "https://yuanlansj.xin/v1",
            "api_key_env": "YUANLAN_API_KEY",
            "context_window": 128000,
            "max_tokens": 8192,
            "cost_tier": "low",
            "modes": {"chat": {}},
        },
        # "qwen": {
        #     "provider": "openai",
        #     "model": "qwen-max",
        #     "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        #     "api_key_env": "QWEN_API_KEY",
        #     "context_window": 128000,
        #     "max_tokens": 8192,
        #     "cost_tier": "low",
        #     "modes": {"chat": {}, "deepthink": {"reasoning_effort": "high"}},
        # },
    },

    # =========================================================
    # Role assignments (static v2)
    #
    # We intentionally pin roles to GPT / Claude / Gemini so the
    # overnight agent workflow has predictable responsibilities.
    # Probing is still useful for availability checks and fallbacks,
    # but explicit model assignment takes precedence over scheduling.
    # =========================================================
    "llm_roles": {
        # --- Analyst layer: low-cost, high-frequency collection ---
        "market_analyst": {
            "model": "gemini",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },
        "news_analyst": {
            "model": "gemini",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },
        "social_analyst": {
            "model": "gemini",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },
        "fundamentals_analyst": {
            "model": "gemini",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },

        # --- Research and synthesis ---
        "bull_researcher": {
            "model": "claude",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },
        "bear_researcher": {
            "model": "claude",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },
        "research_manager": {
            "model": "claude",
            "mode": "deepthink",
            "fallback": [{"model": "gpt", "mode": "deepthink"}],
        },

        # --- Trading decision layer ---
        "trader": {
            "model": "gpt",
            "mode": "deepthink",
            "fallback": [{"model": "claude", "mode": "deepthink"}],
        },
        "portfolio_manager": {
            "model": "gpt",
            "mode": "deepthink",
            "fallback": [{"model": "claude", "mode": "deepthink"}],
        },

        # --- Risk debate: fast, frequent turns ---
        "aggressive_debater": {
            "model": "gemini",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },
        "conservative_debater": {
            "model": "gemini",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },
        "neutral_debater": {
            "model": "gemini",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },

        # --- Post-processing ---
        "reflector": {
            "model": "claude",
            "mode": "deepthink",
            "fallback": [{"model": "gpt", "mode": "deepthink"}],
        },
        "signal_processor": {
            "model": "gemini",
            "mode": "chat",
            "fallback": [{"model": "gpt", "mode": "chat"}],
        },
    },

    # =========================================================
    # Legacy LLM settings (backward compatible fallback)
    # Used when llm_roles is empty or a role is not mapped
    # =========================================================
    "llm_provider": "openai",
    "deep_think_llm": "claude-opus-4-6-thinking",
    "quick_think_llm": "gpt-5.4",
    "backend_url": "http://49.51.249.22/v1",

    # Provider-specific thinking configuration
    "google_thinking_level": None,
    "openai_reasoning_effort": None,
    "anthropic_effort": None,

    # Output language for analyst reports and final decision
    "output_language": "English",

    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,

    # =========================================================
    # Overnight strategy defaults
    # These act as graph/runtime configuration anchors for the
    # upcoming overnight agent integration.
    # =========================================================
    "strategy_mode": "single_stock",
    "overnight_top_n": 5,
    "overnight_exit_mode": "open",
    "overnight_filter_variant": "strict_risk",
    "overnight_weight_variant": "baseline",
    "overnight_candidate_pool_size": 20,
    "overnight_allow_cash": True,
    "overnight_feature_table_path": "data/overnight_mvp/features/overnight_features_20260101_20260430.csv",
    "overnight_labels_path": "data/overnight_labels/csi300_overnight_labels_clean_20240101_20260430.csv",

    # Data vendor configuration
    "data_vendors": {
        "core_stock_apis": "tushare,yfinance",
        "technical_indicators": "tushare,yfinance",
        "fundamental_data": "tushare,yfinance",
        "news_data": "akshare,yfinance",
    },
    "tool_vendors": {},
}

_bridge_openclaw_credentials(DEFAULT_CONFIG)
