# src/predict_file.py
import sys
import argparse
from pathlib import Path
from pipeline import LLMJudgePipeline
from config import SETTINGS


# ── Edit these to test your own inputs ────────────────────────────────────────
INPUTS = [
    {
        "id": "1",
        "text": "The capital of France is Berlin.",
        "question": "What is the capital of France?",
        "knowledge": "France is a country in Western Europe. Its capital is Paris."
    },
    {
        "id": "2",
        "text": "Python was created by Guido van Rossum and first released in 1991.",
        "question": "Who created Python?",
        "knowledge": "Python is a programming language created by Guido van Rossum, first released in 1991."
    }
]
# ─────────────────────────────────────────────────────────────────────────────


def predict_one(pipeline, item_id, text, question="", knowledge=""):
    r         = pipeline.predict(answer=text, question=question, knowledge=knowledge)
    fr        = r["final_result"]
    label_str = "HALLUCINATED" if fr["label"] == 1 else "FACTUAL"
    prob      = fr["probability"]

    RED   = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"
    colour = RED if fr["label"] == 1 else GREEN

    print(f"\n  {'─'*62}")
    print(f"  [{item_id}] {colour}{label_str}{RESET}  (prob={prob:.3f})")
    print(f"  Text     : {text[:100]}")
    if question:
        print(f"  Question : {question[:80]}")
    if knowledge:
        print(f"  Knowledge: {knowledge[:80]}{'...' if len(knowledge) > 80 else ''}")
    print(f"  Strategy : {fr['strategy']}  |  Model: {fr['model']}")


def run_file_mode(pipeline):
    print("\n" + "="*65)
    print("  MODE: File  —  running hardcoded examples")
    print("="*65)
    for item in INPUTS:
        predict_one(pipeline, item["id"], item["text"],
                    item.get("question", ""), item.get("knowledge", ""))
    print(f"\n  {'─'*62}")
    print(f"  Done. {len(INPUTS)} examples processed.")
    pipeline.print_usage()


def run_terminal_mode(pipeline):
    print("\n" + "="*65)
    print("  MODE: Terminal  —  type your own text")
    print("  Type 'quit' or press Enter on an empty line to exit.")
    print("="*65)

    count = 0
    while True:
        print()
        try:
            text = input("  Text to check: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Exiting. Goodbye!\n")
            break

        if text.lower() in ("quit", "exit", "q") or text == "":
            print("\n  Exiting. Goodbye!\n")
            break

        count    += 1
        question  = input("  Question  (Enter to skip): ").strip()
        knowledge = input("  Knowledge (Enter to skip): ").strip()
        predict_one(pipeline, count, text, question, knowledge)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model",    default=None)
    args = parser.parse_args()

    strategy = args.strategy or SETTINGS.prompt.default_strategy

    print(f"\nLoading LLM-as-Judge ({strategy})...")
    pipeline = LLMJudgePipeline(strategy=strategy, provider=args.provider, model=args.model)
    print(f"Model: {pipeline.client.model}  |  Strategy: {strategy}")

    print("\n" + "="*65)
    print("  LLM-AS-JUDGE HALLUCINATION DETECTOR — Predict Mode")
    print("="*65)
    print("\n  Choose mode:")
    print("  [1] Run hardcoded examples from file")
    print("  [2] Type your own text in terminal")
    print("  [3] Both")

    try:
        choice = input("\n  Your choice (1 / 2 / 3): ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = "1"

    if choice == "1":
        run_file_mode(pipeline)
    elif choice == "2":
        run_terminal_mode(pipeline)
    elif choice == "3":
        run_file_mode(pipeline)
        run_terminal_mode(pipeline)
    else:
        print("\n  Invalid choice — running file mode by default.")
        run_file_mode(pipeline)


if __name__ == "__main__":
    main()
