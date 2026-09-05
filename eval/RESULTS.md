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
`index/build_index.py` and `index/retrieve.py` are back on
`BAAI/bge-small-en-v1.5` now.

## Faithfulness (groundedness)

Separate from retrieval quality: given a question, the retrieved chunks, and a
generated answer (`agent/generate.py`, one answer per golden question, saved in
`eval/baseline_answers.json`), is every claim in the answer actually backed by
the sources it cites? Scored by an LLM-as-judge (`eval/faithfulness.py`) that
breaks the answer into individual factual claims and checks each one against
the sources:
- **Mean faithfulness**: average, per answer, of supported claims / total claims.
- **Unsupported claim rate**: unsupported claims / total claims, pooled across all answers.

| Date | Generator | Judge | Answers | Mean faithfulness | Unsupported claim rate | Notes |
|------|-----------|-------|--------:|-------------------:|------------------------:|-------|
| 2026-09-05 | `gpt-5.6-terra` (`agent/generate.py`) | Claude, judged by hand this run (see calibration note) | 29 | 100% | 0.0% (0/131 claims) | Baseline. Every one of 131 claims across all 29 answers checked out against its cited source text: no fabrications, no unsupported figures. This is above the 80-95% usually expected of a naive baseline; see the calibration note below for why, and treat this number as provisional until it's re-run through the real automated judge. |
| 2026-09-05 | `gpt-5.6-terra` (`agent/generate.py`) | `gpt-5.6-terra` (`eval/faithfulness.py`, real API call) | 29 | 99.3% | 0.6% (2/355 claims) | Same baseline answers, re-scored by the actual automated judge instead of a manual read. Lands close to the hand-scored 100% above (within a point), which is a decent sign for that manual pass, though the two aren't independent checks of each other (see calibration note). |
| 2026-09-05 | `agent/agent.py` (decompose -> retrieve -> retry -> verify) | `gpt-5.6-terra` (`eval/faithfulness.py`, real API call) | 29 | 99.6% | 0.7% (3/417 claims) | See "Does the agent help?" below - short answer: no, not measurably, and it costs ~2.9x the tokens. |

**How this run was actually scored, and why that matters:** this baseline was
judged by Claude Code reading each answer against its full source chunks
directly, instead of calling the Anthropic API from `eval/faithfulness.py`
(to avoid spending API credits on a first pass). That means the "judge" and
the "calibration check" were the same model in the same sitting, which is a
real limitation, not independent human labels, and not the same thing as
running `eval/faithfulness.py` for real. Two things worth knowing before
trusting this number:
1. The checking was thorough, not a rubber stamp: every claim's key facts (numbers, named
   entities, causal statements) were matched against the actual source text
   pulled from `chunks.parquet`, not skimmed. A couple of quick keyword
   greps produced false "not found" results that turned out to be present
   once read in full context (e.g. a dollar figure in a financial table with
   no "$" or "million" attached to the cell): a reminder that surface-level
   keyword matching under-counts support; reading the actual passage (which
   is what the real judge prompt does, since it receives full source text)
   does not have that problem.
2. Two claims were judged supported on a "reasonable synthesis" basis rather
   than a single verbatim sentence: one paraphrased "exploiting vulnerabilities
   in third-party infrastructure" as "supply-chain compromises," and one
   combined a rolled-up nine-month statement ("Intelligent Cloud revenue
   increased driven by Azure") with a same-quarter dollar figure from a
   different sentence in the same source chunk. A stricter judge could
   reasonably flag either as a half-point overreach rather than full support.
3. **This has not been calibrated against independent human labels.** Before
   trusting this eval for real decisions (e.g. comparing generators or
   prompts), run `python -m eval.faithfulness` for real (needs
   `OPENAI_API_KEY` in `.env`) and hand-check ~10 of its verdicts yourself
   the way the original plan called for: if you agree with the automated
   judge 8+ times out of 10, trust it; if not, tighten `JUDGE` in
   `eval/faithfulness.py`.

`eval/faithfulness.py` was originally written against the Anthropic API (no
key available in this project), so it now judges with `gpt-5.6-terra` instead
- the same model `agent/generate.py` and `agent/agent.py` use to write the
answers. That is a second, separate limitation on top of the one above: the
judge is the same model family as the generator, which tends to be more
forgiving of that model's own kind of mistake than an independent model would
be. If you get an Anthropic key later, switching the judge back to Claude
would make this a more independent check.

One implementation note worth keeping: `gpt-5.6-terra` is a reasoning model,
so `max_completion_tokens` covers its hidden reasoning tokens as well as the
visible JSON. The first real run of `eval/faithfulness.py` crashed partway
through on one answer with `json.decoder.JSONDecodeError: Expecting value` -
the model had spent its whole 1000-token budget reasoning and returned empty
content. Fixed by raising the cap to 4000 and adding a retry plus a
per-answer try/except in `run()` so one bad response logs an `error` entry
instead of taking down the whole batch.

## Does the agent help?

`agent/agent.py` (decompose the question, retrieve per sub-question, retry if
a sub-question comes up short, draft, then verify the draft against sources)
against the naive single-shot `agent/generate.py`, both over the same 29 golden
questions, both scored by the same judge run:

| System | Mean faithfulness | Unsupported claims | Avg tokens/query |
|---|---:|---:|---:|
| Naive baseline (`agent/generate.py`) | 99.3% | 0.6% (2/355) | 7,484 |
| Agent (`agent/agent.py`) | 99.6% | 0.7% (3/417) | 21,751 |

**The agent does not measurably improve faithfulness, and it costs about 2.9x
the tokens per query.** Be honest about what these numbers can and can't show:
with only 2 and 3 unsupported claims total, the 99.3% vs 99.6% gap is noise,
not a signal - a sample this size can't distinguish "essentially the same" from
"very slightly better." If anything, pooled across all claims the agent's
unsupported *rate* is a hair higher, not lower, despite having an explicit
verify step whose whole job is to strip unsupported claims before returning
the answer. That is not what you'd expect a working verify step to do, so it's
worth asking why rather than reporting a null result and moving on:

- **The agent generates more claims per answer** (417 vs 355 across the same
  29 questions, about 2.1 more per answer) because it deduplicates chunks
  across up to two sub-question retrievals instead of pulling a fixed 5, so it
  has more source material to write from. More claims is more surface area for
  an unsupported one to slip in, even at a similar or better per-claim
  accuracy - which is consistent with what the table actually shows.
- **The verify step is the same model checking its own draft**, not an
  independent reviewer. Self-critique by the same model that wrote the draft
  is a weaker check than an independent one, for the same reason the judge
  being `gpt-5.6-terra` (matching the generator) is a weaker check than an
  independent judge would be. Two of the three unsupported claims the agent
  produced read like slightly-too-confident synthesis across sources rather
  than fabrication (e.g. inferring a causal link - "Medicare funding
  reductions were contributors to higher medical costs" - that's plausible but
  not quite what the cited source states) - exactly the kind of thing a
  same-model verify pass tends to wave through, because it's the sort of
  inference that model would make again if asked to re-check it.
- **Spot-checking the judge's catches suggests they're real, not noise.** One
  baseline miss ("stronger foreign currencies had a favorable... effect on
  Europe sales" attributed to "Q3 2026") pins a fact to the wrong quarter - the
  source ties that Europe effect to the first nine months of 2026, not the
  third quarter specifically. That's a legitimate catch this project's own
  manual read (100% baseline row above) missed, which is a good sign for the
  automated judge's calibration even with the same-model-family caveat.

If you want the agent's extra machinery to actually pay for itself, the
verify step needs to be either a different (ideally independent) model, or
given a harder job than "does this look OK to you" - for example handing it
the same claim-by-claim breakdown `eval/faithfulness.py` produces and having
it act on unsupported claims specifically, rather than a free-form rewrite.
Right now it is 3x the cost for a result indistinguishable from not having it.
