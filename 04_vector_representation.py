"""
04_vector_representation.py
=============================
Converts each chunk's text into a dense vector embedding.

Uses sentence-transformers/all-MiniLM-L6-v2: a small, fast, general-purpose
English embedding model — a good default for a first deployed version of
this project. (Swap EMBEDDING_MODEL_NAME for a multilingual model later if
Arabic-language queries need to be supported directly.)
"""

from importlib import import_module

import numpy as np
from sentence_transformers import SentenceTransformer

_chunking_module = import_module("03_chunking")
CHUNKS = _chunking_module.CHUNKS

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load the embedding model once and reuse it."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)


def embed_chunks(chunks: list) -> np.ndarray:
    texts = [c["search_text"] for c in chunks]
    return embed_texts(texts)


if __name__ == "__main__":
    embeddings = embed_chunks(CHUNKS)
    print(f"Embedded {len(CHUNKS)} chunks into vectors of dimension {embeddings.shape[1]}")
    print(f"Embedding matrix shape: {embeddings.shape}")
