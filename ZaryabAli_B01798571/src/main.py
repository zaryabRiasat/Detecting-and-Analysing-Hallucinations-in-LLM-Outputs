# src/main.py

import sys
import argparse
from pipeline import LLMJudgePipeline
from config import SETTINGS


EXAMPLES = [
    {
        "answer":    "The capital of Pakistan is Islamabad.",
        "question":  "What is the capital of Pakistan?",
        "knowledge": "Pakistan is a country in South Asia. Its capital city is Islamabad.",
    },
    {
        "answer":    "Albert Einstein was a professional footballer who played for Germany.",
        "question":  "Who was Albert Einstein?",
        "knowledge": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    }
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None,
                        choices=["zero_shot", "few_shot", "chain_of_thought", "structured", None])
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model",    default=None)
    args = parser.parse_args()

    strategy = args.strategy or SETTINGS.prompt.default_strategy

    print(f"\nLoading LLM-as-Judge pipeline...")
    pipeline = LLMJudgePipeline(
        strategy = strategy,
        provider = args.provider,
        model    = args.model,
    )

    print(f"\n{'='*70}")
    print(f"  LLM-as-Judge Hallucination Detector — Demo")
    print(f"  Strategy: {strategy.upper()}  |  Model: {pipeline.client.model}")
    print(f"{'='*70}")

    for i, ex in enumerate(EXAMPLES, 1):
        result    = pipeline.predict(**ex)
        fr        = result["final_result"]
        label_str = "HALLUCINATED" if fr["label"] == 1 else "FACTUAL"
        prob      = fr["probability"]

        print(f"\n[{i:02d}] {label_str}  (prob={prob:.3f})")
        print(f"     Q: {ex['question']}")
        print(f"     A: {ex['answer'][:100]}")

    pipeline.print_usage()


if __name__ == "__main__":
    main()
