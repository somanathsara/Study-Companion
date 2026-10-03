"""ChromaDB wrapper. THE CONTRACT between ingestion (A) and RAG (B).

Chunk metadata (all values must be str/int/float/bool for Chroma):
  source_type : "pdf" | "slide" | "video"
  doc_id, doc_name, path
  locator     : human string shown in citations ("Page 12", "Slide 5", "12:35-13:20")
  page        : int (pdf/slide)      start_sec/end_sec : int (video, later)
  section     : nearest heading      topic : topic tag (= section for now)
  has_images  : bool (hook for diagram handling in a later phase)
"""
import chromadb
import config

_collection = None


def get_collection(ef=None):
    """ef is only for tests; in the app Chroma's default embedder (MiniLM, ONNX) is used."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        kwargs = {"embedding_function": ef} if ef is not None else {}
        _collection = client.get_or_create_collection(
            "course", metadata={"hnsw:space": "cosine"}, **kwargs
        )
    return _collection


def add_chunks(chunks: list[dict]) -> None:
    """chunks: [{"id", "text", **metadata}] ; upsert => re-uploading a file never duplicates."""
    col = get_collection()
    for i in range(0, len(chunks), 100):
        batch = chunks[i:i + 100]
        col.upsert(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{k: v for k, v in c.items() if k not in ("id", "text")} for c in batch],
        )


def search(query: str, k: int = config.TOP_K, where: dict | None = None) -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    r = col.query(query_texts=[query], n_results=min(k, col.count()), where=where)
    return [
        {"id": r["ids"][0][i], "text": r["documents"][0][i],
         "meta": r["metadatas"][0][i], "distance": r["distances"][0][i]}
        for i in range(len(r["ids"][0]))
    ]


def list_docs() -> list[dict]:
    col = get_collection()
    metas = col.get(include=["metadatas"])["metadatas"]
    docs = {}
    for m in metas:
        d = docs.setdefault(m["doc_id"], {"doc_id": m["doc_id"], "doc_name": m["doc_name"], "chunks": 0})
        d["chunks"] += 1
    return list(docs.values())


def delete_doc(doc_id: str) -> None:
    get_collection().delete(where={"doc_id": doc_id})
