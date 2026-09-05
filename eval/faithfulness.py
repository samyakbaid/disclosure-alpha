import json
import os
import re
from pathlib import Path

import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

ROOT = Path(__file__).resolve().parents[1]

JUDGE = """You are evaluating whether an answer is grounded in its sources.

Break the ANSWER into individual factual claims. For each claim, decide if it is
directly supported by the SOURCES. Ignore hedging and meta-statements.

Respond with ONLY a JSON object, no other text:
{"total_claims": <int>, "supported_claims": <int>,
 "unsupported": ["claim text", ...]}"""


def score_answer(answer_text, sources):
    src = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(sources, 1))
    msg = client.messages.create(
        model="claude-sonnet-4-6", max_tokens=1000, system=JUDGE,
        messages=[{"role": "user",
                   "content": f"SOURCES:\n{src}\n\nANSWER:\n{answer_text}"}],
    )
    raw = msg.content[0].text.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    d = json.loads(raw)
    d["faithfulness"] = (d["supported_claims"] / d["total_claims"]
                         if d["total_claims"] else 1.0)
    return d


def run(answers_file, out_file=None):
    answers = json.load(open(answers_file))
    chunks = pd.read_parquet(ROOT / "data" / "processed" / "chunks.parquet")
    scores, unsupported_total, claims_total = [], 0, 0
    results = []
    for a in answers:
        srcs = [{"text": chunks[chunks.chunk_id == cid].iloc[0]["text"]}
                for cid in a["source_ids"]
                if len(chunks[chunks.chunk_id == cid])]
        s = score_answer(a["answer"], srcs)
        scores.append(s["faithfulness"])
        claims_total += s["total_claims"]
        unsupported_total += s["total_claims"] - s["supported_claims"]
        results.append({"question": a["question"], **s})
        print(".", end="", flush=True)
    print(f"\nMean faithfulness: {sum(scores) / len(scores):.1%}")
    print(f"Unsupported claim rate: {unsupported_total / claims_total:.1%} "
          f"({unsupported_total}/{claims_total} claims)")
    if out_file:
        json.dump(results, open(out_file, "w"), indent=2)
        print(f"Per-answer results written to {out_file}")


if __name__ == "__main__":
    run(ROOT / "eval" / "baseline_answers.json",
        out_file=ROOT / "eval" / "faithfulness_results.json")
