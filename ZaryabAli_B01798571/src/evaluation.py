# src/evaluation.py
import sys
import json
import time
import argparse
from pathlib import Path
from collections import Counter

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split

from pipeline import LLMJudgePipeline
from config import SETTINGS

def load_dataset(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def run_evaluation(
    dataset_path: str,
    strategy:     str   = None,
    provider:     str   = None,
    model:        str   = None,
    threshold:    float = 0.5,
    n_samples:    int   = 0,
) -> dict:
    strategy = strategy or SETTINGS.prompt.default_strategy
    provider = provider or SETTINGS.api.provider
    model    = model    or SETTINGS.api.models.get(provider)

    data   = load_dataset(dataset_path)
    labels = [r["label"] for r in data]
    _, test = train_test_split(data, test_size=0.01, stratify=labels, random_state=42)

    if n_samples and n_samples < len(test):
        import random; random.seed(42)
        random.shuffle(test)
        test = test[:n_samples]

    pipeline = LLMJudgePipeline(
        strategy  = strategy,
        provider  = provider,
        model     = model,
        threshold = threshold,
    )

    y_true, y_pred, y_scores = [], [], []
    times, error_types = [], []

    print(f"Evaluating {len(test)} samples with strategy={strategy}, model={model}...")
    t_global = time.time()

    for i, row in enumerate(test):
        t0     = time.time()
        result = pipeline.predict_row(row)
        elapsed = time.time() - t0

        pred       = result["label"]
        score      = result["probability"]
        true_label = row["label"]

        y_true.append(true_label)
        y_pred.append(pred)
        y_scores.append(score)
        times.append(elapsed)

        if   true_label == 1 and pred == 1: et = "TP"
        elif true_label == 0 and pred == 0: et = "TN"
        elif true_label == 0 and pred == 1: et = "FP"
        else:                               et = "FN"
        error_types.append(et)

        if (i + 1) % 50 == 0:
            elapsed_total = time.time() - t_global
            print(f"  {i+1}/{len(test)} done  ({elapsed_total:.1f}s)")

        time.sleep(60.0 / SETTINGS.api.requests_per_minute)

    roc = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else None

    metrics = {
        "total_samples":    len(y_true),
        "strategy":         strategy,
        "model":            model,
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score":         round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":          round(roc, 4) if roc else None,
        "avg_inference_ms": round(sum(times) / len(times) * 1000, 2),
        "total_time_s":     round(time.time() - t_global, 2),
    }

    ec = Counter(error_types)
    metrics["TP"] = ec["TP"]; metrics["TN"] = ec["TN"]
    metrics["FP"] = ec["FP"]; metrics["FN"] = ec["FN"]

    per_source = {}
    for row, et in zip(test, error_types):
        src = row.get("source", "unknown")
        per_source.setdefault(src, {"TP": 0, "TN": 0, "FP": 0, "FN": 0})
        per_source[src][et] += 1
    metrics["per_source"] = per_source
    metrics["api_usage"]  = pipeline.usage_summary()

    results_dir = Path("../results")
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(test)
    df["prediction"]  = y_pred
    df["score"]       = y_scores
    df["error_type"]  = error_types
    df.to_csv(results_dir / "predictions.csv", index=False)
    df[df["error_type"].isin(["FP","FN"])].to_csv(results_dir / "errors.csv", index=False)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  EVALUATION RESULTS (LLM-as-Judge — {strategy})")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k not in ("per_source", "api_usage"):
            print(f"  {k:<20}: {v}")
    print(f"\n  TP:{ec['TP']}  TN:{ec['TN']}  FP:{ec['FP']}  FN:{ec['FN']}")
    print("\n  Per-source:")
    for src, counts in per_source.items():
        total = sum(counts.values())
        correct = counts["TP"] + counts["TN"]
        print(f"    {src:<14}: acc={correct/total:.3f}  "
              f"TP={counts['TP']} TN={counts['TN']} FP={counts['FP']} FN={counts['FN']}")
    pipeline.print_usage()
    print("\n  Saved → results/metrics.json, predictions.csv, errors.csv")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="../data/processed/combined_dataset.json")
    parser.add_argument("--strategy",   default=None)
    parser.add_argument("--provider",   default=None)
    parser.add_argument("--model",      default=None)
    parser.add_argument("--threshold",  type=float, default=0.5)
    parser.add_argument("--n-samples",  type=int,   default=0)
    args = parser.parse_args()
    run_evaluation(args.dataset, args.strategy, args.provider, args.model, args.threshold, args.n_samples)
