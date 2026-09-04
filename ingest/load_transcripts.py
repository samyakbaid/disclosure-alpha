from pathlib import Path
from datasets import load_dataset
from ingest.build_corpus import UNIVERSE   # reuse the same 15 tickers

ROOT = Path(__file__).resolve().parents[1]

ds = load_dataset("kurry/sp500_earnings_transcripts", split="train")
mine = ds.filter(lambda x: x["symbol"] in UNIVERSE)
mine.to_parquet(ROOT / "data" / "processed" / "transcripts.parquet")
print(f"{len(mine)} transcripts for your universe")