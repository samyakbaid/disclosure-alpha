from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

ROOT = Path(__file__).resolve().parents[1]
_model = SentenceTransformer("BAAI/bge-m3")
_model.max_seq_length = 1024
_client = QdrantClient(path=str(ROOT / "data" / "qdrant"))

def retrieve(query, k=5, ticker=None, section=None):
    """Find the k chunks most related in meaning to the query.
    Optional filters narrow it to one company or one section."""
    conds = []
    if ticker:  conds.append(FieldCondition(key="ticker",  match=MatchValue(value=ticker)))
    if section: conds.append(FieldCondition(key="section", match=MatchValue(value=section)))
    qv = _model.encode(query, normalize_embeddings=True).tolist()
    resp = _client.query_points(
        collection_name="filings",
        query=qv,
        limit=k,
        query_filter=Filter(must=conds) if conds else None,
    )
    return [{"score": h.score, **h.payload} for h in resp.points]

if __name__ == "__main__":
    for r in retrieve("What are the main risk factors?", ticker="AAPL", k=3):
        print(f"[{r['score']:.2f}] {r['ticker']} {r['section']}: {r['text'][:150]}...")