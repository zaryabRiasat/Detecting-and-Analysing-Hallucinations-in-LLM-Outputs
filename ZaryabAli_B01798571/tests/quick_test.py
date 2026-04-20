# tests/quick_test.py
"""
Quick unit tests for LLM-as-Judge detector.
Tests all non-API components. No API key required.

Run:
    cd tests
    python quick_test.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from utils import normalize_text, truncate, parse_verdict, parse_confidence, cost_estimate
from config import SETTINGS
from prompts.strategies import build_prompt, FEW_SHOT_EXAMPLES

_r = {"passed": 0, "failed": 0}
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(label, condition):
    if condition:
        print(f"  [{PASS}] {label}"); _r["passed"] += 1
    else:
        print(f"  [{FAIL}] {label}"); _r["failed"] += 1


# ─── 1. Utils ─────────────────────────────────────────────────────────────────
def test_utils():
    print("\n[1] Utils")
    check("normalize_text", normalize_text("  hello   world  ") == "hello world")
    check("truncate short",  truncate("short", 100) == "short")
    check("truncate long",   truncate("x" * 200, 50).endswith("..."))
    check("cost_estimate > 0", cost_estimate(1000, 500, "claude-sonnet-4-20250514") > 0)
    check("cost_estimate zero", cost_estimate(0, 0, "claude-sonnet-4-20250514") == 0.0)


# ─── 2. parse_verdict ─────────────────────────────────────────────────────────
def test_parse_verdict():
    print("\n[2] parse_verdict")
    check("JSON hallucinated",  parse_verdict('{"verdict": "HALLUCINATED"}') == 1)
    check("JSON factual",       parse_verdict('{"verdict": "FACTUAL"}')      == 0)
    check("JSON label int 1",   parse_verdict('{"label": 1}')                == 1)
    check("JSON label int 0",   parse_verdict('{"label": 0}')                == 0)
    check("keyword HALLUCINATED", parse_verdict("This is HALLUCINATED.")     == 1)
    check("keyword FACTUAL",      parse_verdict("Answer is FACTUAL.")        == 0)
    check("keyword factual",      parse_verdict("The answer is factual.")    == 0)
    check("VERDICT: HALLUCINATED",parse_verdict("VERDICT: HALLUCINATED")     == 1)
    check("VERDICT: FACTUAL",     parse_verdict("VERDICT: FACTUAL")          == 0)
    check("unsupported → 1",      parse_verdict("This claim is unsupported.") == 1)
    check("unrecognised → None",  parse_verdict("I am not sure.") is None)


# ─── 3. parse_confidence ──────────────────────────────────────────────────────
def test_parse_confidence():
    print("\n[3] parse_confidence")
    check("JSON confidence",  abs(parse_confidence('{"confidence": 0.87}') - 0.87) < 0.01)
    check("JSON probability", abs(parse_confidence('{"probability": 0.92}') - 0.92) < 0.01)
    check("inline pattern",   abs(parse_confidence("confidence: 0.75") - 0.75) < 0.01)
    check("percentage",       abs(parse_confidence("score: 85%") - 0.85) < 0.01)
    check("default 0.5",      parse_confidence("I cannot determine") == 0.5)
    check("clamp to [0,1]",   0.0 <= parse_confidence('{"confidence": 1.5}') <= 1.0)


# ─── 4. Prompting strategies ──────────────────────────────────────────────────
def test_prompts():
    print("\n[4] Prompting Strategies")
    answer    = "The capital of Germany is Madrid."
    question  = "What is the capital of Germany?"
    knowledge = "Germany's capital is Berlin."

    for strategy in ["zero_shot", "few_shot", "chain_of_thought", "structured"]:
        sys_p, usr_p = build_prompt(strategy, answer, question, knowledge)
        check(f"{strategy} returns non-empty system", len(sys_p) > 0)
        check(f"{strategy} returns non-empty user",   len(usr_p) > 0)
        check(f"{strategy} contains answer",          answer[:30] in usr_p or "Madrid" in usr_p)

    # few_shot includes examples
    _, usr_few = build_prompt("few_shot", answer, question, knowledge)
    check("few_shot includes examples", "FACTUAL" in usr_few or "HALLUCINATED" in usr_few)

    # chain_of_thought has step-by-step
    _, usr_cot = build_prompt("chain_of_thought", answer, question, knowledge)
    check("chain_of_thought has steps", "Step" in usr_cot)

    # structured has JSON template
    _, usr_str = build_prompt("structured", answer, question, knowledge)
    check("structured has JSON template", '"verdict"' in usr_str)

    # Invalid strategy raises ValueError
    try:
        build_prompt("invalid_strategy", answer)
        check("invalid strategy raises ValueError", False)
    except ValueError:
        check("invalid strategy raises ValueError", True)


# ─── 5. Config ────────────────────────────────────────────────────────────────
def test_config():
    print("\n[5] Config")
    check("provider defined",         SETTINGS.api.provider in ("anthropic", "openai"))
    check("anthropic model defined",  "anthropic" in SETTINGS.api.models)
    check("openai model defined",     "openai"    in SETTINGS.api.models)
    check("4 strategies defined",     len(SETTINGS.prompt.strategies) == 4)
    check("default strategy valid",   SETTINGS.prompt.default_strategy in SETTINGS.prompt.strategies)
    check("few_shot_k > 0",           SETTINGS.prompt.few_shot_k > 0)
    check("max_tokens > 0",           SETTINGS.api.max_tokens > 0)
    check("temperature == 0",         SETTINGS.api.temperature == 0.0)


# ─── 6. Few-shot examples ─────────────────────────────────────────────────────
def test_few_shot_examples():
    print("\n[6] Few-Shot Example Bank")
    check("5 examples defined",  len(FEW_SHOT_EXAMPLES) >= 5)
    for ex in FEW_SHOT_EXAMPLES:
        check(f"example has answer",  "answer"  in ex)
        check(f"example has verdict", "verdict" in ex)
        check(f"verdict is valid",    ex["verdict"] in ("FACTUAL", "HALLUCINATED"))


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  LLM-as-Judge: Quick Tests")
    print("=" * 55)

    test_utils()
    test_parse_verdict()
    test_parse_confidence()
    test_prompts()
    test_config()
    test_few_shot_examples()

    total = _r["passed"] + _r["failed"]
    print(f"\n{'='*55}")
    print(f"  Results: {_r['passed']}/{total} passed")
    print("  All tests passed!" if _r["failed"] == 0 else f"  {_r['failed']} test(s) failed.")
    print("=" * 55)
    sys.exit(0 if _r["failed"] == 0 else 1)
