import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from index.retrieve import retrieve

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

CHEAP = "gpt-5.6-luna"     # $0.20/$1.20 per 1M: decompose, sufficiency checks
STRONG = "gpt-5.6-terra"   # $2/$12 per 1M: synthesis, verification


def _call(model, system, prompt, max_completion_tokens=1000):
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_completion_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content, resp.usage


def decompose(question):
    """Split into sub-questions. Cheap model."""
    sys = ("Split the question into 1-3 standalone sub-questions, each "
           "answerable from a single filing section. Respond with ONLY a JSON "
           'array of strings: ["sub-question 1", ...]')
    text, usage = _call(CHEAP, sys, question, max_completion_tokens=300)
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(text), usage
    except Exception:
        return [question], usage


def sufficient(subq, chunks):
    """Does this context answer the sub-question? Cheap model."""
    ctx = "\n\n".join(c["text"][:600] for c in chunks)
    sys = ("Answer with exactly one word, YES or NO: do the excerpts contain "
           "enough information to answer the question?")
    text, usage = _call(CHEAP, sys, f"Question: {subq}\n\nExcerpts:\n{ctx}",
                        max_completion_tokens=10)
    return text.strip().upper().startswith("YES"), usage


SYNTH = """You are a financial analyst assistant. Answer using ONLY the excerpts.
- Every claim must be supported by an excerpt.
- Cite inline as [1], [2] matching excerpt numbers.
- If the excerpts don't answer it, say so. Never guess."""

VERIFY = """Check each claim in the ANSWER against the SOURCES. Rewrite the
answer removing any claim not directly supported. Keep citations intact.
Respond with ONLY the corrected answer text."""


def run(question, ticker=None, k=5, verbose=True):
    trace, tokens_in, tokens_out = [], 0, 0

    subqs, u = decompose(question)
    tokens_in += u.prompt_tokens; tokens_out += u.completion_tokens
    trace.append({"step": "decompose", "output": subqs})

    all_chunks, seen = [], set()
    for sq in subqs:
        got = retrieve(sq, k=k, ticker=ticker)
        ok, u = sufficient(sq, got)
        tokens_in += u.prompt_tokens; tokens_out += u.completion_tokens
        if not ok:
            got += retrieve(f"{sq} details specifics", k=k, ticker=ticker)
            trace.append({"step": "retry", "subq": sq})
        for c in got:
            if c["chunk_id"] not in seen:
                seen.add(c["chunk_id"]); all_chunks.append(c)
        trace.append({"step": "retrieve", "subq": sq, "n": len(got)})

    ctx = "\n\n".join(
        f"[{i}] {c['ticker']} | {c['fiscal_period']} | {c['section']}\n{c['text']}"
        for i, c in enumerate(all_chunks, 1))

    draft, u = _call(CHEAP, SYNTH, f"Excerpts:\n\n{ctx}\n\nQuestion: {question}")
    tokens_in += u.prompt_tokens; tokens_out += u.completion_tokens

    final, u = _call(CHEAP, VERIFY, f"SOURCES:\n{ctx}\n\nANSWER:\n{draft}")
    tokens_in += u.prompt_tokens; tokens_out += u.completion_tokens
    trace.append({"step": "verify", "changed": final.strip() != draft.strip()})

    if verbose:
        for t in trace:
            print(f"  · {t}")

    return {"answer": final, "sources": all_chunks, "trace": trace,
            "usage": {"in": tokens_in, "out": tokens_out}}


if __name__ == "__main__":
    r = run("How does Apple describe supply chain risk, and what does it say "
            "about foreign exchange exposure?", ticker="AAPL")
    print("\n" + r["answer"])
    print(f"\ntokens: {r['usage']}")