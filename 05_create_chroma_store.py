"""
05_create_chroma_store.py
============================
Builds a persistent ChromaDB collection from the chunk embeddings.

Run this once (or whenever 01_documents.py changes) to (re)build the
vector store on disk at ./chroma_db. The Streamlit app only reads from
this store at request time — it does not rebuild it on every query.
"""

from importlib import import_module

import chromadb

_chunking_module = import_module("03_chunking")
_vector_module = import_module("04_vector_representation")

CHUNKS = _chunking_module.CHUNKS
embed_chunks = _vector_module.embed_chunks

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "finance_knowledge"


def create_chroma_store(chunks: list, db_path: str = CHROMA_DB_PATH,
                         collection_name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=db_path)

    # Rebuild fresh each run so re-running after editing 01_documents.py
    # never leaves stale or duplicate chunks behind.
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.create_collection(collection_name)

    embeddings = embed_chunks(chunks)

    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings.tolist(),
        documents=[c["chunk_text"] for c in chunks],
        metadatas=[
            {
                "document_id": c["document_id"],
                "title": c["title"],
                "doc_type": c["doc_type"],
                "effective_date": c["effective_date"],
                "is_current": c["is_current"],
            }
            for c in chunks
        ],
    )
    return collection


if __name__ == "__main__":
    collection = create_chroma_store(CHUNKS)
    print(f"Stored {collection.count()} chunks in ChromaDB collection "
          f"'{COLLECTION_NAME}' at '{CHROMA_DB_PATH}'")
