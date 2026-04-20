# src/api_client.py
"""
LLM API Client
==============
Handles API calls to Anthropic (Claude).

Features:
  - Automatic retry with exponential backoff
  - Rate limiting
  - Token usage and cost tracking
  - Structured error handling and logging
"""

import os
import time
import json
import logging
from typing import Optional, Tuple, Dict, Any

from config import SETTINGS
from utils import cost_estimate, rate_limit_sleep

logger = logging.getLogger(__name__)


class APIError(Exception):
    pass


class LLMClient:

    def __init__(
        self,
        provider:   str   = None,
        model:      str   = None,
        max_tokens: int   = None,
        temperature: float = None,
        max_retries: int  = 3,
        retry_delay: float = 2.0,
    ):
        self.provider    = provider    or SETTINGS.api.provider
        self.model       = model       or SETTINGS.api.models.get(self.provider, "claude-sonnet-4-6")
        self.max_tokens  = max_tokens  or SETTINGS.api.max_tokens
        self.temperature = temperature if temperature is not None else SETTINGS.api.temperature
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Usage tracking
        self.total_calls         = 0
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd      = 0.0

        # Validate API key exists
        self._check_api_key()

    def _check_api_key(self) -> None:
        if self.provider == "anthropic":
            key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not key:
                logger.warning(
                    "ANTHROPIC_API_KEY not set. "
                    "Export it before running: export ANTHROPIC_API_KEY=sk-ant-..."
                )
        elif self.provider == "openai":
            key = os.environ.get("OPENAI_API_KEY", "")
            if not key:
                logger.warning(
                    "OPENAI_API_KEY not set. "
                    "Export it before running: export OPENAI_API_KEY=sk-..."
                )

    def call(
        self,
        system_prompt: str,
        user_prompt:   str,
    ) -> Tuple[str, Dict[str, Any]]:
        
        last_error = None

        for attempt in range(1, self.max_retries + 1):
            try:
                if self.provider == "anthropic":
                    text, usage = self._call_anthropic(system_prompt, user_prompt)
                elif self.provider == "openai":
                    text, usage = self._call_openai(system_prompt, user_prompt)
                else:
                    raise APIError(f"Unknown provider: {self.provider}")

                # Track usage
                self.total_calls         += 1
                self.total_prompt_tokens += usage.get("prompt_tokens", 0)
                self.total_output_tokens += usage.get("output_tokens", 0)
                self.total_cost_usd      += usage.get("cost_usd", 0.0)

                return text, usage

            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    wait = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(f"API call failed (attempt {attempt}/{self.max_retries}): {e}. "
                                   f"Retrying in {wait:.1f}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"All {self.max_retries} attempts failed: {e}")

        raise APIError(f"API call failed after {self.max_retries} attempts: {last_error}")

    def _call_anthropic(
        self,
        system_prompt: str,
        user_prompt:   str,
    ) -> Tuple[str, Dict]:
        try:
            import anthropic
        except ImportError:
            raise APIError("anthropic library not installed. Run: pip install anthropic")

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        message = client.messages.create(
            model       = self.model,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
            system      = system_prompt,
            messages    = [{"role": "user", "content": user_prompt}],
        )

        text = message.content[0].text if message.content else ""
        prompt_tokens = message.usage.input_tokens
        output_tokens = message.usage.output_tokens
        cost = cost_estimate(prompt_tokens, output_tokens, self.model)

        return text, {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      cost,
        }

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt:   str,
    ) -> Tuple[str, Dict]:
        """Call OpenAI GPT-4 API."""
        try:
            from openai import OpenAI
        except ImportError:
            raise APIError("openai library not installed. Run: pip install openai")

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model       = self.model,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
            messages    = [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
        )

        text          = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost          = cost_estimate(prompt_tokens, output_tokens, self.model)

        return text, {
            "prompt_tokens": prompt_tokens,
            "output_tokens": output_tokens,
            "cost_usd":      cost,
        }

    def usage_summary(self) -> Dict[str, Any]:
        return {
            "total_calls":         self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd":      round(self.total_cost_usd, 4),
            "avg_cost_per_call":   round(
                self.total_cost_usd / max(self.total_calls, 1), 6
            ),
        }

    def print_usage(self) -> None:
        u = self.usage_summary()
        print(f"\n  API Usage Summary:")
        print(f"    Calls:          {u['total_calls']}")
        print(f"    Prompt tokens:  {u['total_prompt_tokens']:,}")
        print(f"    Output tokens:  {u['total_output_tokens']:,}")
        print(f"    Estimated cost: ${u['total_cost_usd']:.4f} USD")
        print(f"    Avg per call:   ${u['avg_cost_per_call']:.6f} USD")
