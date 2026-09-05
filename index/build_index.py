from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ROOT = Path(__file__).resolve().parents[1]

# Upgraded from bge-small-en-v1.5 to the bigger bge-m3 (1024-dim). See
# eval/RESULTS.md: this made retrieval worse, not better. Best guess why -
# bge-m3's own default max_seq_length is 8192, but it's capped to 1024 below
# and most of our chunks run ~1700 tokens, so this truncates most chunks
# before they're even embedded.
_model = SentenceTransformer("BAAI/bge-m3")
_model.max_seq_length = 1024

chunks = pd.read_parquet(ROOT / "data" / "processed" / "chunks.parquet")
vecs = _model.encode(chunks["text"].tolist(), batch_size=32,
                     show_progress_bar=True, normalize_embeddings=True)

client = QdrantClient(path=str(ROOT / "data" / "qdrant"))   # a local file-based DB
if client.collection_exists("filings"):
    client.delete_collection("filings")
client.create_collection(
    "filings",
    vectors_config=VectorParams(size=vecs.shape[1], distance=Distance.COSINE),
)
client.upsert("filings", [
    PointStruct(id=i, vector=vecs[i].tolist(),
                payload=chunks.iloc[i].to_dict())
    for i in range(len(chunks))
])
print(f"Indexed {len(chunks)} chunks.")