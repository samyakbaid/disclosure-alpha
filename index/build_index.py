from pathlib import Path
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ROOT = Path(__file__).resolve().parents[1]

# Upgraded from bge-small-en-v1.5 to the bigger bge-m3 (1024-dim). See
# eval/RESULTS.md: on this corpus it's worse, not better, and it's a ranking
# problem, not a config bug. The right chunk is retrievable in the top 50 for
# every golden question either way; bge-m3 just ranks it much lower (11-45)
# than bge-small does (top 3-5). A max_seq_length cap was tried and dropped
# below since it didn't meaningfully change the numbers either way.
device = ("cuda" if torch.cuda.is_available()
          else "mps" if torch.backends.mps.is_available()
          else "cpu")
_model = SentenceTransformer("BAAI/bge-m3", device=device)
print(f"Using device: {device}")   # so you can confirm in the job log

chunks = pd.read_parquet(ROOT / "data" / "processed" / "chunks.parquet")
vecs = _model.encode(chunks["text"].tolist(), batch_size=64,
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
