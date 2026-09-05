# DisclosureAlpha

A little RAG system over SEC filings. It pulls the last few 10-K and 10-Q
filings for 15 big companies, splits the narrative parts into chunks, embeds
them, and lets you ask questions and get back the relevant passages.

Right now it covers: AAPL, MSFT, NVDA, JPM, XOM, PG, UNH, HD, CAT, KO, PFE, BA,
DIS, CSCO, INTC.

## How it fits together

```
ingest/edgar.py         talks to the SEC EDGAR API
ingest/build_corpus.py  downloads the filings -> data/processed/manifest.parquet
ingest/chunk.py         pulls out Business / Risk Factors / MD&A / Market Risk
                        and splits them -> data/processed/chunks.parquet
index/build_index.py    embeds the chunks (bge-small-en-v1.5) into a local
                        Qdrant db at data/qdrant
index/retrieve.py       semantic search over the index
agent/generate.py       answers a question: retrieves chunks, asks an LLM to
                        answer using only those chunks, cites them inline
eval/                   a hand-labeled question set (Recall@k / MRR) and a
                        faithfulness/groundedness check on generated answers,
                        both tracked in eval/RESULTS.md
```

Everything regenerable (the filings, the chunks, the vector db) lives under
`data/` and is gitignored, so a fresh clone has to build it from scratch.

## Setup

You need Python 3.12ish and about 1 GB of disk for the data.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The SEC wants you to identify yourself, so make a `.env` file (it is gitignored,
do not commit it):

```
SEC_USER_AGENT="Your Name project-name your@email.com"
```

## Running the pipeline

Run everything from the project root with `python -m folder.file`, not by cd-ing
into the folders. Order matters, each step reads the previous step's output.

```bash
python -m ingest.build_corpus     # download filings (a few minutes, be nice to SEC)
python -m ingest.chunk            # build chunks.parquet
python -m index.build_index       # embed + index (downloads the model first time)
python -m index.retrieve          # quick sanity check, prints a few results
```

Then the eval side:

```bash
python -m eval.build_golden       # interactive, label the seed questions
python -m eval.run                # prints Recall@3/5/10 and MRR
```

`eval.build_golden` shows you the top candidates for each seed question and you
pick the chunk that actually answers it (or skip if nothing does). It appends to
`eval/golden.jsonl` so you can stop and come back.

Heads up: every time you re-run `ingest.chunk` the chunk ids are regenerated
(they are fresh UUIDs), so the golden labels point at ids that no longer exist.
If you change the chunking, re-run `index.build_index` and then re-label.

Then, to check the generated answers are actually grounded in their sources
(not just that retrieval found the right chunks):

```bash
python -m agent.generate           # needs OPENAI_API_KEY in .env; try one question
python -m eval.faithfulness        # needs ANTHROPIC_API_KEY in .env; scores a batch
```

`eval.faithfulness` expects a file of generated answers (question, answer,
source_ids) at `eval/baseline_answers.json`; there is no script yet that
generates that file in bulk from `eval/golden.jsonl`, so build one (a loop over
`agent.generate`) before running it fresh. It calls the Anthropic API once per
answer to judge groundedness, so it costs real API credits: see the calibration
note below before you run a large batch.

Optional: `python -m ingest.load_transcripts` grabs earnings call transcripts for
the same companies into `data/processed/transcripts.parquet`. Nothing uses them
yet.

## Calibrating the faithfulness judge

`eval/faithfulness.py` is an LLM-as-judge: it reads an answer and its cited
sources and decides how many of the answer's claims are actually backed by
them. An LLM judge is only worth trusting once you know it agrees with a human
reading the same material, so before you rely on its numbers, hand-check
around 10 of its verdicts yourself: read the answer, read the sources, decide
if you agree. Aim for 8 or more out of 10 agreeing; if it is lower, tighten the
`JUDGE` prompt in `eval/faithfulness.py` and check again.

The baseline logged in `eval/RESULTS.md` (100% faithfulness, 29 answers, 131
claims) was not scored this way. To avoid spending API credits on a first
pass, Claude Code read every answer against its full source text directly
instead of calling the Anthropic API, so the same model produced both the
verdicts and the "calibration" check in one sitting. That is a real limitation
worth knowing about (it is not independent human labels), even though the
checking itself was thorough. Read the calibration note in `eval/RESULTS.md`
for what that run actually found before treating 100% as a settled number.
Once you run `eval.faithfulness` for real, replace this baseline with an
actual hand-calibrated run and note the agreement rate here, for example:
"LLM-as-judge calibrated against 10 human labels, 90% agreement."

## Notes on coverage

`ingest/chunk.py` tries to be robust to how differently each company formats its
filings (all caps headers, headers stuck inside tables, 10-K vs 10-Q item
numbers, cross-references that look like headers). A couple of known gaps:

- INTC lays its whole 10-K out in tables, so stripping the number tables also
  strips the section headers. It currently produces no chunks.
- XOM only has a 10-Q in the corpus (no 10-K), so there is no Business section
  and its Risk Factors / Market Risk are just "see our 10-K" pointers.
- Market Risk is empty for a few companies (CAT, PG, JPM 10-K) because they fold
  it into MD&A by reference instead of writing a separate section. That is
  expected, not a bug.
