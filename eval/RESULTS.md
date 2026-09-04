# Evaluation Results

Retrieval quality on the golden question set. One row per run; append new rows as
the pipeline changes (embedding model, chunking, corpus, retrieval params).

Metrics are computed by `python -m eval.run` against `eval/golden.jsonl`:
- **Recall@k**: fraction of questions where a labeled supporting chunk appears in the top *k*.
- **MRR**: mean reciprocal rank of the first labeled supporting chunk.

| Date | Model | Corpus | Golden n | Recall@3 | Recall@5 | Recall@10 | MRR | Notes |
|------|-------|--------|---------:|---------:|---------:|----------:|-----:|-------|
| 2026-09-04 | `BAAI/bge-small-en-v1.5` | 1,637 chunks · 14 companies · SEC 10-K/10-Q | 29 | 75.9% | 89.7% | 89.7% | 0.606 | Baseline. Recall@5 == Recall@10 indicates ~10% of questions are hard misses (correct chunk not in top 10), not ranking failures. |
