# src/cost_analysis.py

import json
import argparse
import time
from pathlib import Path
from utils import cost_estimate


# ── Rough prompt length estimates (characters) ────────────────────────────────
STRATEGY_OVERHEAD = {
    "zero_shot":        120,
    "few_shot":         900,   # 3 examples ≈ 300 chars each
    "chain_of_thought": 350,
    "structured":       280,
}

AVG_KNOWLEDGE_CHARS = 300
AVG_ANSWER_CHARS    = 80
AVG_RESPONSE_CHARS  = {
    "zero_shot":        10,    # single word
    "few_shot":         12,
    "chain_of_thought": 200,   # step-by-step reasoning
    "structured":       120,   # JSON object
}

CHARS_PER_TOKEN = 4.0   # rough approximation


def estimate_tokens(strategy: str) -> dict:
    prompt_chars = (
        STRATEGY_OVERHEAD[strategy]
        + AVG_KNOWLEDGE_CHARS
        + AVG_ANSWER_CHARS
        + 150   # system prompt
    )
    completion_chars = AVG_RESPONSE_CHARS[strategy]

    return {
        "strategy":          strategy,
        "prompt_tokens":     int(prompt_chars / CHARS_PER_TOKEN),
        "completion_tokens": int(completion_chars / CHARS_PER_TOKEN),
    }


def cost_for_n_samples(strategy: str, n: int, model: str) -> dict:
    est = estimate_tokens(strategy)
    total_prompt     = est["prompt_tokens"]     * n
    total_completion = est["completion_tokens"] * n
    total_cost       = cost_estimate(total_prompt, total_completion, model)

    return {
        "strategy":           strategy,
        "n_samples":          n,
        "model":              model,
        "prompt_tokens_each": est["prompt_tokens"],
        "completion_tokens_each": est["completion_tokens"],
        "total_prompt_tokens":    total_prompt,
        "total_completion_tokens": total_completion,
        "estimated_cost_usd": round(total_cost, 4),
        "cost_per_sample_usd": round(total_cost / n, 6),
    }


def print_cost_table(model: str, n_samples: int) -> None:
    print(f"\n  Cost Estimates  |  Model: {model}  |  Samples: {n_samples}")
    print(f"  {'Strategy':<20} {'Prompt tok':>12} {'Comp tok':>10} "
          f"{'Cost/sample':>13} {'Total cost':>12}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*13} {'-'*12}")

    for strategy in ["zero_shot", "few_shot", "chain_of_thought", "structured"]:
        c = cost_for_n_samples(strategy, n_samples, model)
        print(f"  {strategy:<20} {c['prompt_tokens_each']:>12} "
              f"{c['completion_tokens_each']:>10} "
              f"${c['cost_per_sample_usd']:>12.6f} "
              f"${c['estimated_cost_usd']:>11.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       default="claude-sonnet-4-6")
    parser.add_argument("--n-samples",   type=int, default=1000)
    parser.add_argument("--run-sample",  type=int, default=0,
                        help="If > 0, run this many live API calls per strategy")
    args = parser.parse_args()

    print("=" * 65)
    print("  LLM-as-Judge — Cost & Efficiency Analysis")
    print("=" * 65)

    print_cost_table(args.model, args.n_samples)

    # Also show for mini model
    mini_model = "claude-haiku-4-5-20251001"
    print_cost_table(mini_model, args.n_samples)

    print(f"\n  Recommendation:")
    print(f"  - Use 'structured' for production (predictable output, moderate cost)")
    print(f"  - Use 'zero_shot' for large-scale screening (cheapest)")
    print(f"  - Use 'chain_of_thought' for difficult borderline cases (most accurate)")
    print(f"  - Use Haiku model to reduce cost by ~10x vs Sonnet")

    # Live sample if requested
    if args.run_sample > 0:
        from pipeline import LLMJudgePipeline
        import random
        with open("../data/processed/combined_dataset.json") as f:
            data = json.load(f)
        sample = random.sample(data, min(args.run_sample, len(data)))

        results = {}
        for strategy in ["zero_shot", "structured"]:
            pl = LLMJudgePipeline(strategy=strategy)
            times = []
            for row in sample:
                t0 = time.time()
                pl.predict_row(row)
                times.append((time.time() - t0) * 1000)
                time.sleep(1.0)
            results[strategy] = {
                "avg_ms":   round(sum(times) / len(times), 1),
                "usage":    pl.usage_summary(),
            }

        print("\n  Live Sample Results:")
        for s, r in results.items():
            print(f"  {s}: avg={r['avg_ms']}ms  cost=${r['usage']['total_cost_usd']:.4f}")

        Path("../results").mkdir(parents=True, exist_ok=True)
        with open("../results/cost_analysis.json", "w") as f:
            json.dump(results, f, indent=2)
        print("  Saved → results/cost_analysis.json")


if __name__ == "__main__":
    main()
