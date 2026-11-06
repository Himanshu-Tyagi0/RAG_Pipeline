import json
import hashlib
from RAG_full_pipeline import VectorStoreManager


def content_id(text: str) -> str:
    """Stable ID based on text content — reindex pe bhi same rahega."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def dump_chunks(out_file="chunks_dump.json"):
    vs = VectorStoreManager()
    data = vs.collection.get(include=["documents", "metadatas"])

    chunks = []
    for text, meta in zip(data["documents"], data["metadatas"]):
        chunks.append({
            "id": content_id(text),
            "preview": text[:200].replace("\n", " "),
            "full_text": text,
            "metadata": meta,
        })

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"✓ Wrote {len(chunks)} chunks to {out_file}")
    print(f"  Read this file, note down IDs of chunks that answer your test queries.")


if __name__ == "__main__":
    dump_chunks()