import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ingest.edgar import recent_filings, _get

# The project root is the folder that contains "ingest".
# This file is at disclosure-alpha/ingest/build_corpus.py, so
# parents[1] walks up two levels to disclosure-alpha/.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["AAPL", "MSFT", "NVDA", "JPM", "XOM", "PG", "UNH",
            "HD", "CAT", "KO", "PFE", "BA", "DIS", "CSCO", "INTC"]

rows = []
for tkr in tqdm(UNIVERSE):
    for f in recent_filings(tkr, n=8):
        dest = RAW / f"{tkr}_{f['form']}_{f['reportDate']}.html"
        if not dest.exists():
            dest.write_bytes(_get(f["url"]).content)
        f["ticker"] = tkr
        f["local_path"] = str(dest)
        rows.append(f)

manifest = pd.DataFrame(rows)
manifest.to_parquet(PROCESSED / "manifest.parquet", index=False)
print(f"{len(manifest)} filings across {manifest.ticker.nunique()} tickers")