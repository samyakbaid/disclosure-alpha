# Evaluation Results

Retrieval quality on the golden question set. One row per run; append new rows as
the pipeline changes (embedding model, chunking, corpus, retrieval params).

Metrics are computed by `python -m eval.run` against `eval/golden.jsonl`:
- **Recall@k**: fraction of questions where a labeled supporting chunk appears in the top *k*.
- **MRR**: mean reciprocal rank of the first labeled supporting chunk.

| Date | Model | Corpus | Golden n | Recall@3 | Recall@5 | Recall@10 | MRR | Notes |
|------|-------|--------|---------:|---------:|---------:|----------:|-----:|-------|
| 2026-09-04 | `BAAI/bge-small-en-v1.5` | 1,637 chunks · 14 companies · SEC 10-K/10-Q | 29 | 75.9% | 89.7% | 89.7% | 0.606 | Baseline. Recall@5 == Recall@10 indicates ~10% of questions are hard misses (correct chunk not in top 10), not ranking failures. |
| 2026-09-05 | `BAAI/bge-m3`, `max_seq_length=1024` | same corpus, re-embedded | 29 | 24.1% | 44.8% | 62.1% | 0.245 | Worse across the board. At the time this looked like it might be chunk truncation (chunks run ~1700 tokens, well past the 1024 cap), so the cap was dropped and it was re-run, see the row below. |
| 2026-09-05 | `BAAI/bge-m3`, no seq-length cap (run on HPC, H100) | same corpus, re-embedded | 29 | 20.7% | 51.7% | 58.6% | 0.224 | Dropping the cap did not fix it, ruling out truncation as the cause. Dug further: for every golden question the right chunk IS in the top 50, it is just ranked way lower with bge-m3 (rank 11 to 45) than with bge-small (top 3 to 5). So this is bge-m3's dense embedding ranking worse on this task, not a bug in our code. Likely explanation: bge-m3 is built for hybrid dense + sparse + multi-vector retrieval and broad multilingual coverage, using only its dense vector (which is all a plain cosine-similarity Qdrant index does) is not its strong suit, while bge-small-en-v1.5 was trained specifically to be a strong standalone dense retriever for English text like this. |

## bge-small vs bge-m3, side by side

|  | bge-small-en-v1.5 | bge-m3 (capped) | bge-m3 (uncapped) |
|---|---:|---:|---:|
| Recall@3 | 75.9% | 24.1% | 20.7% |
| Recall@5 | 89.7% | 44.8% | 51.7% |
| Recall@10 | 89.7% | 62.1% | 58.6% |
| MRR | 0.606 | 0.245 | 0.224 |

Bottom line: on these numbers, bge-small-en-v1.5 is the better choice for this
project as it stands. bge-m3 is not broken, the answer chunk is always
somewhere in its results, it just does not rank it highly with a plain dense
cosine search. It would likely do much better in a hybrid setup (dense + its
own sparse scores), but that is a bigger change than swapping a model name.
The code currently still points at bge-m3 (`index/build_index.py`,
`index/retrieve.py`); switch it back to `BAAI/bge-small-en-v1.5` and re-run
`index.build_index` if you want the better numbers in production.
