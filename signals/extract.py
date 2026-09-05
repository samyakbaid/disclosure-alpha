import os, json, re
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
ROOT = Path(__file__).resolve().parents[1]

PROMPT = """Compare two versions of the same filing section from consecutive
periods. Identify substantive changes only — ignore rewording.

Respond with ONLY JSON:
{"new_topics": ["..."], "removed_topics": ["..."],
 "tone_shift": "more_negative"|"neutral"|"more_positive",
 "severity": 0-5}"""

def diff_section(ticker, section, text_old, text_new):
    resp = client.chat.completions.create(
        model="gpt-5.6-terra",
       max_completion_tokens=800,
        messages=[
            {"role": "system", "content": PROMPT},
            {"role": "user", "content":
             f"EARLIER:\n{text_old[:6000]}\n\nLATER:\n{text_new[:6000]}"},
        ],
    )
    raw = re.sub(r"^```(json)?|```$", "",
                 resp.choices[0].message.content.strip(), flags=re.M)
    return json.loads(raw.strip())

def main():
    chunks = pd.read_parquet(ROOT/"data"/"processed"/"chunks.parquet")
    # join chunks back into full sections per (ticker, period, section)
    secs = (chunks.sort_values("chunk_id")
            .groupby(["ticker","form","fiscal_period","section"])["text"]
            .apply(" ".join).reset_index())

    rows = []
    for (tkr, sec), grp in secs.groupby(["ticker","section"]):
        grp = grp.sort_values("fiscal_period")
        periods = grp.to_dict("records")
        for prev, curr in zip(periods, periods[1:]):
            try:
                d = diff_section(tkr, sec, prev["text"], curr["text"])
            except Exception as e:
                print(f"skip {tkr} {sec} {curr['fiscal_period']}: {e}")
                continue
            rows.append({
                "ticker": tkr, "section": sec,
                "fiscal_period": curr["fiscal_period"],
                "prev_period": prev["fiscal_period"],
                "n_new": len(d.get("new_topics", [])),
                "n_removed": len(d.get("removed_topics", [])),
                "tone_shift": d.get("tone_shift"),
                "severity": d.get("severity", 0),
                "new_topics": json.dumps(d.get("new_topics", [])),
            })
            print(f"{tkr} {sec} {curr['fiscal_period']}: "
                  f"+{rows[-1]['n_new']} sev={rows[-1]['severity']}")

    df = pd.DataFrame(rows)
    df.to_parquet(ROOT/"data"/"processed"/"signals.parquet", index=False)
    print(f"\n{len(df)} signal rows")

if __name__ == "__main__":
    main()