# src/prompt_comparison.py

import json
import time
import argparse
import logging
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

from pipeline import LLMJudgePipeline
from utils import normalize_text
from config import SETTINGS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def run_strategy(
    strategy:   str,
    samples:    list,
    provider:   str,
    model:      str,
) -> dict:
    pipeline = LLMJudgePipeline(
        strategy  = strategy,
        provider  = provider,
        model     = model,
    )

    y_true, y_pred, y_scores = [], [], []
    rows_out = []
    total_cost = 0.0
    times = []

    print(f"\n  Running strategy: {strategy.upper()} ({len(samples)} samples)...")

    for i, row in enumerate(samples):
        try:
            result = pipeline.predict_row(row)
        except Exception as e:
            logger.warning(f"  Sample {i} failed: {e} — skipping")
            continue

        y_true.append(row["label"])
        y_pred.append(result["label"])
        y_scores.append(result["probability"])
        times.append(result.get("inference_ms", 0))
        total_cost += result.get("usage", {}).get("cost_usd", 0.0)

        rows_out.append({
            "id":          row.get("id", i),
            "source":      row.get("source", ""),
            "true_label":  row["label"],
            "pred_label":  result["label"],
            "probability": result["probability"],
            "strategy":    strategy,
        })

        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{len(samples)} done  cost_so_far=${total_cost:.4f}")

        # Rate limiting
        time.sleep(60.0 / SETTINGS.api.requests_per_minute)

    # Compute metrics
    if not y_true:
        return {"strategy": strategy, "error": "no samples processed"}, []

    roc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else None
    metrics = {
        "strategy":        strategy,
        "n_samples":       len(y_true),
        "accuracy":        round(accuracy_score(y_true, y_pred), 4),
        "precision":       round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":          round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":              round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":         round(roc, 4) if roc else None,
        "avg_inference_ms": round(sum(times) / max(len(times), 1), 2),
        "total_cost_usd":  round(total_cost, 4),
        "cost_per_sample": round(total_cost / max(len(y_true), 1), 6),
    }

    print(f"  [{strategy}] Acc={metrics['accuracy']}  F1={metrics['f1']}  "
          f"Cost=${metrics['total_cost_usd']:.4f}")

    return metrics, rows_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="../data/processed/combined_dataset.json")
    parser.add_argument("--strategy",   default="all",
                        choices=["all", "zero_shot", "few_shot", "chain_of_thought", "structured"])
    parser.add_argument("--n-samples",  type=int, default=100)
    parser.add_argument("--provider",   default=None)
    parser.add_argument("--model",      default=None)
    args = parser.parse_args()

    provider = args.provider or SETTINGS.api.provider
    model    = args.model    or SETTINGS.api.models.get(provider)

    print("=" * 60)
    print("  Prompt Strategy Comparison")
    print(f"  Provider: {provider}  |  Model: {model}")
    print("=" * 60)

    # Load + sample data
    data   = load_dataset(args.dataset)
    labels = [r["label"] for r in data]
    _, val = train_test_split(data, test_size=0.01, stratify=labels, random_state=42)

    # Balanced sample
    hall  = [r for r in val if r["label"] == 1][:args.n_samples // 2]
    fact  = [r for r in val if r["label"] == 0][:args.n_samples // 2]
    sample = hall + fact
    print(f"\nSample: {len(sample)} ({len(hall)} hallucinated, {len(fact)} factual)")

    strategies = (
        SETTINGS.prompt.strategies
        if args.strategy == "all"
        else [args.strategy]
    )

    all_metrics = []
    all_rows    = []

    for strategy in strategies:
        metrics, rows = run_strategy(strategy, sample, provider, model)
        all_metrics.append(metrics)
        all_rows.extend(rows)

    # Save results
    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "prompt_comparison.json", "w", encoding='utf-8') as f:
        json.dump(all_metrics, f, indent=2)

    if all_rows:
        pd.DataFrame(all_rows).to_csv(results_dir / "prompt_comparison.csv", index=False)

    # Print comparison table
    print(f"\n{'='*60}")
    print("  STRATEGY COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Strategy':<20} {'Acc':>7} {'F1':>7} {'AUC':>7} {'Cost/sample':>12}")
    print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7} {'-'*12}")
    for m in all_metrics:
        if "error" in m:
            continue
        auc = f"{m['roc_auc']:.4f}" if m.get("roc_auc") else "N/A"
        print(f"  {m['strategy']:<20} {m['accuracy']:>7.4f} {m['f1']:>7.4f} "
              f"{auc:>7} ${m['cost_per_sample']:>10.6f}")

    best = max((m for m in all_metrics if "error" not in m), key=lambda x: x["f1"], default=None)
    if best:
        print(f"\n  Best strategy: {best['strategy']} (F1={best['f1']})")
    print(f"\n  Results saved → results/prompt_comparison.json")


if __name__ == "__main__":
    main()