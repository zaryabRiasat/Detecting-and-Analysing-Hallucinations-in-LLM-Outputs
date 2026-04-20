# src/build_dataset.py
"""
build_dataset.py — Dataset Builder
Shared dataset for all four group members.
"""
import json, random
from pathlib import Path
import pandas as pd

INCLUDE_FEVER = False
RANDOM_SEED   = 42

def load_truthfulqa_csv(path):
    df = pd.read_csv(path)
    rows = []
    for i, row in df.iterrows():
        q = str(row.get("Question","")).strip()
        a = str(row.get("Best Answer","")).strip()
        if q and a:
            rows.append({"id":f"truthfulqa_{i}","source":"TruthfulQA","task_type":"qa",
                         "text":a,"label":0,"hallucination_type":"none","question":q,"knowledge":""})
    return rows

def load_fever_jsonl(path):
    rows = []
    with open(path,"r",encoding="utf-8") as f:
        for i,line in enumerate(f):
            line=line.strip()
            if not line: continue
            item=json.loads(line)
            label=str(item.get("label","")).strip().upper()
            claim=str(item.get("claim","")).strip()
            if not claim: continue
            if label=="SUPPORTS": mapped=0
            elif label=="REFUTES": mapped=1
            else: continue
            rows.append({"id":f"fever_{i}","source":"FEVER","task_type":"claim","text":claim,
                         "label":mapped,"hallucination_type":"factual" if mapped==1 else "none",
                         "question":"","knowledge":""})
    return rows

def load_halueval_jsonl(path):
    rows = []
    with open(path,"r",encoding="utf-8") as f:
        for i,line in enumerate(f):
            line=line.strip()
            if not line: continue
            try: item=json.loads(line)
            except: continue
            knowledge=str(item.get("knowledge","")).strip()
            question=str(item.get("question","")).strip()
            right=str(item.get("right_answer","")).strip()
            hall=str(item.get("hallucinated_answer","")).strip()
            if right:
                rows.append({"id":f"halueval_right_{i}","source":"HaluEval","task_type":"qa",
                             "text":right,"label":0,"hallucination_type":"none",
                             "question":question,"knowledge":knowledge})
            if hall:
                rows.append({"id":f"halueval_hall_{i}","source":"HaluEval","task_type":"qa",
                             "text":hall,"label":1,"hallucination_type":"factual",
                             "question":question,"knowledge":knowledge})
    return rows

def balance_dataset(rows):
    random.seed(RANDOM_SEED)
    hall=[r for r in rows if r["label"]==1]
    truth=[r for r in rows if r["label"]==0]
    if not hall or not truth: return rows
    n=min(len(hall),len(truth))
    balanced=random.sample(hall,n)+random.sample(truth,n)
    random.shuffle(balanced)
    return balanced

if __name__=="__main__":
    all_rows=[]
    tqa=load_truthfulqa_csv("../data/raw/truthfulqa/TruthfulQA.csv")
    all_rows.extend(tqa); print(f"TruthfulQA: {len(tqa)} rows")
    if INCLUDE_FEVER:
        fever=load_fever_jsonl("../data/raw/fever/train.jsonl")
        all_rows.extend(fever); print(f"FEVER: +{len(fever)} rows")
    else:
        print("FEVER: skipped")
    halu=load_halueval_jsonl("../data/raw/halueval/qa_data.json")
    all_rows.extend(halu); print(f"HaluEval: +{len(halu)} rows (total: {len(all_rows)})")
    all_rows=balance_dataset(all_rows)
    Path("../data/processed").mkdir(parents=True,exist_ok=True)
    with open("../data/processed/combined_dataset.json","w",encoding="utf-8") as f:
        json.dump(all_rows,f,indent=2,ensure_ascii=False)
    labels=[r["label"] for r in all_rows]
    print(f"\nSaved {len(all_rows)} rows → data/processed/combined_dataset.json")
    print(f"  Factual (0):      {labels.count(0)}")
    print(f"  Hallucinated (1): {labels.count(1)}")
