import json
from pathlib import Path
import pandas as pd
from index.retrieve import retrieve
from eval.seed_questions import SEED

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = pd.read_parquet(ROOT / "data" / "processed" / "chunks.parquet")
GOLDEN = ROOT / "eval" / "golden.jsonl"

def already_done() -> set:
    if not GOLDEN.exists():
        return set()
    with open(GOLDEN) as f:
        return {json.loads(l)["question"] for l in f if l.strip()}
def snippet(text, kw=None, width=170):
    """Show text around the matched keyword, not just the chunk's start."""
    if kw:
        i = text.lower().find(kw.lower())
        if i != -1:
            start = max(0, i - 70)
            return "..." + text[start:i + 110].strip() + "..."
    return text[:width].strip() + "..."

def show(df, kw=None):
    for i, (_, r) in enumerate(df.iterrows()):
        print(f"  [{i}] {r['chunk_id'][:8]}  {r['ticker']} | {r['section']} | {r['fiscal_period']}")
        print(f"       {snippet(r['text'], kw)}")

def label(q):
    print("\n" + "=" * 74)
    print(f"Q: {q['question']}")
    print(f"   (expected company={q['ticker']}, section≈{q['section']})")
    cands = pd.DataFrame(retrieve(q["question"], k=8, ticker=q["ticker"]))
    print("\nSemantic candidates:")
    show(cands)
    print("\nType the [number] of the chunk that best answers it,")
    print("  'k <word>' to keyword-search this company's chunks,")
    print("  's' to skip (unanswerable/not captured), 'q' to save & quit.")

    pool = cands
    while True:
        choice = input("> ").strip()
        if choice == "q":
            return "quit"
        if choice == "s":
            return None
        if choice.startswith("k "):
            kw = choice[2:].strip().lower()
            hits = CHUNKS[(CHUNKS.ticker == q["ticker"]) &
                          (CHUNKS.text.str.lower().str.contains(kw))].head(8)
            if len(hits):
                pool = hits.reset_index(drop=True)
                print(f"\nKeyword '{kw}' matches:")
                show(pool, kw)
            else:
                print(f"  no chunks contain '{kw}' for {q['ticker']}")
            continue
        if choice.isdigit() and int(choice) < len(pool):
            row = pool.iloc[int(choice)]
            return {"question": q["question"], "ticker": q["ticker"],
                    "section": q["section"],
                    "supporting_chunk_ids": [row["chunk_id"]]}
        print("  didn't understand that — try again")

def main():
    done = already_done()
    todo = [q for q in SEED if q["question"] not in done]
    print(f"{len(done)} already labeled, {len(todo)} to go.")
    with open(GOLDEN, "a") as f:
        for q in todo:
            result = label(q)
            if result == "quit":
                break
            if result is None:
                continue
            f.write(json.dumps(result) + "\n")
            f.flush()
            print("  ✓ saved")
    print(f"\nDone. Golden set: {GOLDEN}")

if __name__ == "__main__":
    main()