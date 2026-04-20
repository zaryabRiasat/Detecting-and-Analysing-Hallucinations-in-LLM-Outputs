# Zaryab Ali B01798571 — LLM-as-Judge Hallucination Detector
**MSc Project | University of the West of Scotland**
**Detecting and Analysing Hallucinations in Large Language Model Outputs**

---

## What This Project Does

This system detects hallucinations in AI-generated text using the **LLM-as-Judge** paradigm.
It sends text to Claude AI via API and asks it to judge whether the text is factually correct or hallucinated.

No model training is required — the system uses Claude's built-in knowledge directly.

---

## Four Prompting Strategies

| Strategy | Description | Best For |
|---|---|---|
| `zero_shot` | Direct question, no examples | Large-scale screening |
| `few_shot` | 3 labelled examples provided | Consistent baseline |
| `chain_of_thought` | Step-by-step reasoning | Difficult cases |
| `structured` | JSON output with confidence score | Production pipelines |

---

## Project Structure

```
zaryabali_b01798571/
│
├── src/                        # Main source code
│   ├── main.py                 # Demo — run 2 example predictions
│   ├── evaluation.py           # Full evaluation — generates results
│   ├── prompt_comparison.py    # Compare all 4 strategies
│   ├── cost_analysis.py        # Token usage and cost estimates
│   ├── predict_file.py         # Interactive prediction tool
│   ├── pipeline.py             # Core prediction pipeline
│   ├── api_client.py           # Claude API client
│   ├── config.py               # Configuration settings
│   ├── utils.py                # Helper functions
│   ├── build_dataset.py        # Dataset builder
│   └── prompts/
│       └── strategies.py       # All 4 prompting strategies
│
├── data/
│   ├── raw/
│   │   ├── truthfulqa/         # TruthfulQA dataset (817 questions)
│   │   ├── halueval/           # HaluEval dataset (QA pairs)
│   │   └── fever/              # FEVER dataset (fact-checking)
│   └── processed/
│       └── combined_dataset.json  # Combined + balanced dataset
│
├── results/                    # Evaluation outputs saved here
├── tests/
│   └── quick_test.py           # 62 unit tests (no API needed)
│
└── requirements.txt            # Python dependencies
```

---

## Setup

**Step 1 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 2 — Set API key**
```bash
# Windows PowerShell
$env:ANTHROPIC_API_KEY="sk-ant-api03-..."

# Mac / Linux
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

---

## How to Run

**1. Run tests (no API key needed)**
```bash
cd tests
python quick_test.py
# Expected: 62/62 passed
```

**2. Run demo**
```bash
cd src
python main.py
```

**3. Run full evaluation (generates dissertation results)**
```bash
cd src
python evaluation.py --n-samples 100 --strategy zero_shot
```

**4. Compare all 4 strategies**
```bash
cd src
python prompt_comparison.py --n-samples 100
```

**5. Cost analysis**
```bash
cd src
python cost_analysis.py --run-sample 10 --strategy zero_shot
```

---

## Datasets

| Dataset | Size | Task |
|---|---|---|
| TruthfulQA | 817 questions | Misconception detection |
| HaluEval | 10,000 QA pairs | Hallucination detection |
| FEVER | 185,000 claims | Fact verification |

---

## Evaluation Metrics

- **Accuracy** — Overall correct predictions
- **Precision** — Of predicted hallucinations, how many were correct
- **Recall** — Of actual hallucinations, how many were detected
- **F1 Score** — Balance of precision and recall
- **ROC-AUC** — Overall detection quality

---


## Model

- **Provider:** Anthropic
- **Model:** claude-sonnet-4-6
- **Temperature:** 0.0 (deterministic)
- **Max tokens:** 512
