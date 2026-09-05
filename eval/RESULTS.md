# Evaluation Results

Retrieval quality on the golden question set. One row per run; append new rows as
the pipeline changes (embedding model, chunking, corpus, retrieval params).

Metrics are computed by `python -m eval.run` against `eval/golden.jsonl`:
- **Recall@k**: fraction of questions where a labeled supporting chunk appears in the top *k*.
- **MRR**: mean reciprocal rank of the first labeled supporting chunk.

| Date | Model | Corpus | Golden n | Recall@3 | Recall@5 | Recall@10 | MRR | Notes |
|------|-------|--------|---------:|---------:|---------:|----------:|-----:|-------|
| 2026-09-04 | `BAAI/bge-small-en-v1.5` | 1,637 chunks · 14 companies · SEC 10-K/10-Q | 29 | 75.9% | 89.7% | 89.7% | 0.606 | Baseline. Recall@5 == Recall@10 indicates ~10% of questions are hard misses (correct chunk not in top 10), not ranking failures. |
| 2026-09-05 | `BAAI/bge-m3` | 1,637 chunks · 14 companies · SEC 10-K/10-Q (same corpus, re-embedded) | 29 | 24.1% | 44.8% | 62.1% | 0.245 | Worse across the board, not a mixed result. Likely cause: the code caps `max_seq_length` to 1024, but bge-m3 defaults to 8192 and our chunks run ~1700 tokens on average (87% of a sample exceed 1024), so most chunks are silently truncated before they're embedded. Would want to re-run without that cap before concluding bge-m3 itself is worse than bge-small. |

## bge-small vs bge-m3, side by side

|  | bge-small-en-v1.5 | bge-m3 | change |
|---|---:|---:|---:|
| Recall@3 | 75.9% | 24.1% | -51.8 pts |
| Recall@5 | 89.7% | 44.8% | -44.9 pts |
| Recall@10 | 89.7% | 62.1% | -27.6 pts |
| MRR | 0.606 | 0.245 | -0.361 |

Switching to the bigger model made every metric worse on this corpus, it did not
just move the trade-offs around. Given the truncation issue above, this isn't a
fair fight yet, so don't read it as "bge-m3 is a bad model," read it as "this
config of bge-m3 is worse than bge-small," until it's re-tested without the
1024 cap.
