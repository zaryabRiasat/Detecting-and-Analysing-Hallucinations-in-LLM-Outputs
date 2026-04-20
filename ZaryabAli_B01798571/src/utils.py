# src/utils.py
import re
import time
import json
import random
from typing import Any, Optional


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def truncate(text: str, max_chars: int = 800) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 3] + "..."


def parse_verdict(response_text: str) -> Optional[int]:

    text = response_text.strip()

    # 1. Try JSON parsing
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            # Check common verdict keys
            for key in ("verdict", "label", "result", "classification", "is_hallucinated"):
                val = obj.get(key)
                if val is None:
                    continue
                if isinstance(val, int):
                    return 1 if val == 1 else 0
                val_str = str(val).lower().strip()
                if val_str in ("hallucinated", "hallucination", "1", "true", "yes", "incorrect", "false_claim"):
                    return 1
                if val_str in ("factual", "correct", "0", "false", "no", "supported", "true_claim"):
                    return 0
        except (json.JSONDecodeError, ValueError):
            pass

    # 2. Keyword scan (case-insensitive, whole word preferred)
    text_lower = text.lower()
    # Strong hallucination signals
    for phrase in ["hallucinated", "hallucination", "not factual", "incorrect", "false claim",
                   "unsupported", "fabricated", "verdict: 1", "label: 1"]:
        if phrase in text_lower:
            return 1
    # Strong factual signals
    for phrase in ["factual", "correct", "supported", "true claim", "verdict: 0", "label: 0",
                   "not hallucinated", "no hallucination"]:
        if phrase in text_lower:
            return 0

    # 3. Yes/No signals
    if re.search(r'\byes\b', text_lower):
        return 1   # "yes, this is hallucinated"
    if re.search(r'\bno\b', text_lower):
        return 0

    return None   # unparseable


def parse_confidence(response_text: str) -> float:
    
    text = response_text.strip()

    # JSON
    json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group())
            for key in ("confidence", "probability", "score", "certainty"):
                val = obj.get(key)
                if val is not None:
                    try:
                        f = float(val)
                        return max(0.0, min(1.0, f))
                    except (TypeError, ValueError):
                        pass
        except (json.JSONDecodeError, ValueError):
            pass

    # Look for "confidence: 0.87" or "score: 85%" patterns
    patterns = [
        r'confidence[:\s]+([0-9]*\.?[0-9]+)',
        r'probability[:\s]+([0-9]*\.?[0-9]+)',
        r'score[:\s]+([0-9]*\.?[0-9]+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            try:
                val = float(m.group(1))
                if val > 1.0:
                    val /= 100.0   # percentage
                return max(0.0, min(1.0, val))
            except ValueError:
                pass

    return 0.5


def rate_limit_sleep(requests_per_minute: int) -> None:
    if requests_per_minute > 0:
        time.sleep(60.0 / requests_per_minute)


def cost_estimate(
    prompt_tokens:     int,
    completion_tokens: int,
    model:             str,
) -> float:
    # Per 1M tokens (input / output)
    pricing = {
        "claude-sonnet-4-6":        (3.0,  15.0),
        "claude-sonnet-4-20250514": (3.0,  15.0),   # legacy alias
        "claude-haiku-4-5-20251001": (0.25, 1.25),
        "gpt-4o":                   (5.0,  15.0),
        "gpt-4o-mini":              (0.15,  0.60),
    }
    inp, out = pricing.get(model, (5.0, 15.0))
    return (prompt_tokens * inp + completion_tokens * out) / 1_000_000
