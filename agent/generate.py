import os
from dotenv import load_dotenv
from openai import OpenAI
from index.retrieve import retrieve

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

SYSTEM = """You are a financial analyst assistant. Answer questions using ONLY
the provided excerpts from SEC filings. Rules:
- Every claim must be supported by an excerpt.
- Cite sources inline as [1], [2] matching the excerpt numbers.
- If the excerpts don't contain the answer, say so plainly. Do not guess.
- Be concise and specific. Quote figures exactly as they appear."""

def format_context(chunks):
    """Turn retrieved chunks into a numbered block the model can cite."""
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f"[{i}] {c['ticker']} | {c['form']} | {c['fiscal_period']} | "
            f"{c['section']}\n{c['text']}"
        )
    return "\n\n".join(parts)

def answer(question, ticker=None, section=None, k=5):
    chunks = retrieve(question, k=k, ticker=ticker, section=section)
    if not chunks:
        return {"answer": "No relevant filings found.", "sources": []}

    resp = client.chat.completions.create(
        model="gpt-5.6-terra",
        max_completion_tokens=1000,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content":
             f"Excerpts:\n\n{format_context(chunks)}\n\nQuestion: {question}"},
        ],
    )
    return {
        "answer": resp.choices[0].message.content,
        "sources": chunks,
        "usage": {"in": resp.usage.prompt_tokens,
                  "out": resp.usage.completion_tokens},
    }

if __name__ == "__main__":
    r = answer("What risks does Apple face from third-party manufacturing?",
               ticker="AAPL")
    print(r["answer"])
    print("\n--- sources ---")
    for i, s in enumerate(r["sources"], 1):
        print(f"[{i}] {s['ticker']} {s['section']} {s['fiscal_period']}")
    print(f"\ntokens: {r['usage']}")
