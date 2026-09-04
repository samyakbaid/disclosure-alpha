from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

ROOT = Path(__file__).resolve().parents[1]

# Start small/fast to get the pipeline working. Upgrade later to
# "BAAI/bge-m3" for higher quality once everything runs end to end.
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

chunks = pd.read_parquet(ROOT / "data" / "processed" / "chunks.parquet")
vecs = model.encode(chunks["text"].tolist(), batch_size=32,
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