"""
03_chunking.py
================
Splits each preprocessed document into retrieval-sized chunks.

Each document here is already short (a single FAQ answer or guide
paragraph), so a moderate word-based sliding window with overlap is
enough — no document needs to be split into more than 2-3 chunks.
"""

from importlib import import_module

_preprocessing_module = import_module("02_preprocessing")
CLEANED_DOCUMENTS = _preprocessing_module.CLEANED_DOCUMENTS


def chunk_text(text: str, chunk_size: int = 60, overlap: int = 15) -> list:
    """Split text into overlapping word-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_chunks(documents: list) -> list:
    """Turn every document into one or more chunk records with metadata."""
    chunk_records = []
    for doc in documents:
        for chunk_index, chunk_body in enumerate(chunk_text(doc["text"])):
            chunk_records.append({
                "chunk_id": f"doc{doc['document_id']}_chunk{chunk_index}",
                "document_id": doc["document_id"],
                "title": doc["title"],
                "doc_type": doc["doc_type"],
                "effective_date": doc["effective_date"],
                "is_current": doc["is_current"],
                "chunk_index": chunk_index,
                "chunk_text": chunk_body,
                # search_text includes the title so keyword/embedding search
                # can also match on the document's subject, not just its body
                "search_text": f"{doc['title']}. {chunk_body}",
            })
    return chunk_records


CHUNKS = build_chunks(CLEANED_DOCUMENTS)


if __name__ == "__main__":
    print(f"Built {len(CHUNKS)} chunks from {len(CLEANED_DOCUMENTS)} documents")
    multi_chunk_docs = {}
    for c in CHUNKS:
        multi_chunk_docs[c["document_id"]] = multi_chunk_docs.get(c["document_id"], 0) + 1
    docs_split_further = {k: v for k, v in multi_chunk_docs.items() if v > 1}
    print(f"Documents split into more than one chunk: {docs_split_further if docs_split_further else 'none — all documents fit in a single chunk'}")
