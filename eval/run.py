import json
from pathlib import Path
from index.retrieve import retrieve

ROOT = Path(__file__).resolve().parents[1]

def load_golden():
    path = ROOT / "eval" / "golden.jsonl"
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def evaluate(golden, k=5):
    recall_hits, rr_total = 0, 0.0
    for ex in golden:
        gold = set(ex["supporting_chunk_ids"])
        ids = [r["chunk_id"] for r in retrieve(ex["question"], k=k, ticker=ex["ticker"])]
        if gold & set(ids):
            recall_hits += 1
        for rank, cid in enumerate(ids, start=1):   # reciprocal rank
            if cid in gold:
                rr_total += 1.0 / rank
                break
    n = len(golden)
    return recall_hits / n, rr_total / n, n

if __name__ == "__main__":
    g = load_golden()
    if not g:
        print("No golden questions yet. Run: python -m eval.build_golden")
    else:
        for k in (3, 5, 10):
            recall, mrr, n = evaluate(g, k=k)
            print(f"k={k:<2} Recall@{k}={recall:.1%}  MRR={mrr:.3f}  (n={n})")