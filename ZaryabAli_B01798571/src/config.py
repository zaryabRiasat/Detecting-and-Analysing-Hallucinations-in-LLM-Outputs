# src/config.py

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class APIConfig:
    # Provider: "anthropic" or "openai"
    provider: str = "anthropic"

    # Model IDs
    models: Dict[str, str] = field(default_factory=lambda: {
        "anthropic": "claude-sonnet-4-6",
        "openai":    "gpt-4o",
    })

    # Request settings
    max_tokens:   int   = 512
    temperature:  float = 0.0
    timeout_s:    int   = 30

    # Rate limiting / cost control
    requests_per_minute: int   = 60
    max_cost_usd:        float = 5.0   # safety cap
    batch_size:          int   = 50    # samples per run before progress save


@dataclass
class PromptConfig:
    # Available strategies (all implemented)
    strategies: List[str] = field(default_factory=lambda: [
        "zero_shot",
        "few_shot",
        "chain_of_thought",
        "structured",
    ])
    default_strategy: str = "structured"

    few_shot_k: int = 3

    # Self-consistency: number of votes (1 = disabled)
    consistency_votes: int = 1


@dataclass
class EvalConfig:
    # Splits match group standard (60/20/20)
    test_size:  float = 0.20
    random_seed: int  = 42

    # Threshold for binary label from probability
    threshold: float = 0.5


@dataclass
class Settings:
    api:    APIConfig    = field(default_factory=APIConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
    eval:   EvalConfig   = field(default_factory=EvalConfig)
    debug:  bool         = False


SETTINGS = Settings()
