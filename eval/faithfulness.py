import json
import os
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Same strong model agent/generate.py and agent/agent.py use for synthesis.
# Note: that means the generator and the judge are the same model family here
# (no Anthropic key available) - weaker than judging with an independent
# model, since a model is more likely to accept its own kind of mistake.
JUDGE_MODEL = "gpt-5.6-terra"

ROOT = Path(__file__).resolve().parents[1]

JUDGE = """You are evaluating whether an answer is grounded in its sources.

Break the ANSWER into individual factual claims. For each claim, decide if it is
directly supported by the SOURCES. Ignore hedging and meta-statements.

Respond with ONLY a JSON object, no other text:
{"total_claims": <int>, "supported_claims": <int>,
 "unsupported": ["claim text", ...]}"""


def _judge_once(src, answer_text):
    resp = client.chat.completions.create(
        # gpt-5.6-terra is a reasoning model: max_completion_tokens covers its
        # internal reasoning tokens too, not just the visible JSON. 1000 was
        # too tight - it silently ate the whole budget on reasoning for one
        # answer and returned empty content. 4000 leaves real headroom.
        model=JUDGE_MODEL, max_completion_tokens=4000,
        messages=[
            {"role": "system", "content": JUDGE},
            {"role": "user",
             "content": f"SOURCES:\n{src}\n\nANSWER:\n{answer_text}"},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)  # raises on empty/malformed content


def score_answer(answer_text, sources, retries=2):
    src = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(sources, 1))
    for attempt in range(retries + 1):
        try:
            d = _judge_once(src, answer_text)
            break
        except (json.JSONDecodeError, KeyError):
            if attempt == retries:
                raise
    d["faithfulness"] = (d["supported_claims"] / d["total_claims"]
                         if d["total_claims"] else 1.0)
    return d


def run(answers_file, out_file=None):
    answers = json.load(open(answers_file))
    chunks = pd.read_parquet(ROOT / "data" / "processed" / "chunks.parquet")
    scores, unsupported_total, claims_total, failed = [], 0, 0, 0
    results = []
    for a in answers:
        srcs = [{"text": chunks[chunks.chunk_id == cid].iloc[0]["text"]}
                for cid in a["source_ids"]
                if len(chunks[chunks.chunk_id == cid])]
        try:
            s = score_answer(a["answer"], srcs)
        except (json.JSONDecodeError, KeyError) as e:
            # Don't let one bad judge response sink the whole batch - flag it
            # and keep going; it's excluded from the aggregate stats below.
            failed += 1
            results.append({"question": a["question"], "error": str(e)})
            print("x", end="", flush=True)
            continue
        scores.append(s["faithfulness"])
        claims_total += s["total_claims"]
        unsupported_total += s["total_claims"] - s["supported_claims"]
        results.append({"question": a["question"], **s})
        print(".", end="", flush=True)
    print(f"\nMean faithfulness: {sum(scores) / len(scores):.1%} (n={len(scores)})")
    print(f"Unsupported claim rate: {unsupported_total / claims_total:.1%} "
          f"({unsupported_total}/{claims_total} claims)")
    if failed:
        print(f"{failed} answer(s) the judge failed to score after retries - see 'error' entries in the output")
    if out_file:
        json.dump(results, open(out_file, "w"), indent=2)
        print(f"Per-answer results written to {out_file}")


if __name__ == "__main__":
    run(ROOT / "eval" / "baseline_answers.json",
        out_file=ROOT / "eval" / "faithfulness_results.json")
