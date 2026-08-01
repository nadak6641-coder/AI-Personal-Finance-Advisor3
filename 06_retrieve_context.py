"""
06_retrieve_context.py
========================
Retrieval + context building — the most critical stage in this pipeline,
since a weak retriever or a careless context package directly produces a
wrong or hallucinated final answer, no matter how good the prompt is.

Two things happen here, and they are kept as separate steps on purpose:

  1. Retrieval: get a ranked list of candidate chunks (hybrid: semantic
     search via the persisted Chroma store + lexical TF-IDF search,
     blended together). Retrieval does not know what "good" means beyond
     similarity — it will happily return an outdated chunk if it is
     lexically or semantically close to the query.

  2. Context building: turn those candidates into the actual text block
     sent to the LLM. This is where "current beats outdated", "no
     duplicate/near-duplicate chunks", "cap chunks per document" and "stay
     within a word budget" are enforced. A prompt is only as grounded as
     the context package feeding it.

Also included: a ground-truth query set (13+ queries, including
paraphrased wording, an exact-detail query, and a multi-document query)
and an evaluation routine — run this file directly to see retrieval
quality metrics and inspect a handful of full context packages, including
a current-vs-outdated conflict and an out-of-scope query.
"""

import re
from importlib import import_module

import numpy as np
import pandas as pd
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_chunking_module = import_module("03_chunking")
_vector_module = import_module("04_vector_representation")
_preprocessing_module = import_module("02_preprocessing")

CHUNKS = _chunking_module.CHUNKS
embed_texts = _vector_module.embed_texts
normalize_lexical_text = _preprocessing_module.normalize_lexical_text

CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "finance_knowledge"

chunks_df = pd.DataFrame(CHUNKS)


# ---------------------------------------------------------------------------
# Lexical (TF-IDF) retrieval
# ---------------------------------------------------------------------------

_tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))
_tfidf_matrix = _tfidf_vectorizer.fit_transform(chunks_df["search_text"].map(normalize_lexical_text))


def retrieve_lexical(query: str, k: int = 5) -> pd.DataFrame:
    query_vector = _tfidf_vectorizer.transform([normalize_lexical_text(query)])
    scores = cosine_similarity(query_vector, _tfidf_matrix).flatten()
    ranking = np.argsort(scores)[::-1][:k]
    results = chunks_df.iloc[ranking].copy()
    results["score"] = scores[ranking]
    return results


# ---------------------------------------------------------------------------
# Semantic retrieval (queries the persisted Chroma vector store)
# ---------------------------------------------------------------------------

_chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

try:
    _collection = _chroma_client.get_collection(COLLECTION_NAME)
except chromadb.errors.NotFoundError:
    _collection = _chroma_client.create_collection(COLLECTION_NAME)
    # هنا لازم تضيف الـ chunks والـ embeddings فعليًا
    embeddings = embed_texts(chunks_df["search_text"].tolist())
    _collection.add(
        ids=chunks_df["chunk_id"].tolist(),
        embeddings=embeddings.tolist(),
        documents=chunks_df["chunk_text"].tolist() if "chunk_text" in chunks_df else chunks_df["search_text"].tolist(),
        metadatas=chunks_df[["document_id", "title", "doc_type", "effective_date", "is_current"]].to_dict("records"),
    )


# ---------------------------------------------------------------------------
# Hybrid retrieval: blend lexical + semantic scores
# ---------------------------------------------------------------------------

def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.array(scores, dtype=float)
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return np.zeros_like(scores)
    return (scores - lo) / (hi - lo)


def retrieve_hybrid(query: str, alpha: float = 0.6, k: int = 5, pool_size: int = 10) -> pd.DataFrame:
    """
    alpha controls the semantic/lexical blend: alpha=1.0 is pure semantic,
    alpha=0.0 is pure lexical. Both retrievers are run over a larger pool
    (pool_size) first, so a chunk that ranks well in only one method still
    gets a fair combined score instead of being dropped before blending.
    """
    lexical = retrieve_lexical(query, k=pool_size).set_index("chunk_id")
    semantic = retrieve_semantic(query, k=pool_size).set_index("chunk_id")

    all_ids = set(lexical.index) | set(semantic.index)
    rows = []
    for chunk_id in all_ids:
        lexical_score = lexical.loc[chunk_id, "score"] if chunk_id in lexical.index else 0.0
        semantic_score = semantic.loc[chunk_id, "score"] if chunk_id in semantic.index else 0.0
        source_row = semantic.loc[chunk_id] if chunk_id in semantic.index else lexical.loc[chunk_id]
        rows.append({
            "chunk_id": chunk_id, "document_id": source_row["document_id"],
            "title": source_row["title"], "doc_type": source_row["doc_type"],
            "effective_date": source_row["effective_date"], "is_current": source_row["is_current"],
            "chunk_text": source_row["chunk_text"],
            "lexical_score": lexical_score, "semantic_score": semantic_score,
        })

    result = pd.DataFrame(rows)
    result["score"] = (
        alpha * _min_max_normalize(result["semantic_score"].values)
        + (1 - alpha) * _min_max_normalize(result["lexical_score"].values)
    )
    return result.sort_values("score", ascending=False).head(k).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Context building: candidates -> a clean, grounded, current-preferring,
# word-budgeted context block
# ---------------------------------------------------------------------------

def build_context_package(query: str, retrieval_k: int = 8, alpha: float = 0.6,
                           max_context_chunks: int = 3, max_chunks_per_document: int = 1,
                           word_budget: int = 180, prefer_current: bool = True,
                           min_score_ratio: float = 0.40, min_absolute_score: float = 0.05) -> dict:
    candidates = retrieve_hybrid(query, alpha=alpha, k=retrieval_k)

    if prefer_current:
        candidates = candidates.sort_values(
            by=["is_current", "score", "effective_date"], ascending=[False, False, False]
        ).reset_index(drop=True)

    max_score = candidates["score"].max() if len(candidates) else 0.0
    selected, seen_texts, per_doc_count, used_words = [], set(), {}, 0

    for _, row in candidates.iterrows():
        if row["score"] < min_absolute_score:
            continue
        if max_score > 0 and row["score"] < max_score * min_score_ratio:
            continue
        normalized = re.sub(r"\s+", " ", row["chunk_text"]).strip().lower()
        if normalized in seen_texts:
            continue
        if per_doc_count.get(row["document_id"], 0) >= max_chunks_per_document:
            continue
        chunk_words = len(row["chunk_text"].split())
        if selected and used_words + chunk_words > word_budget:
            continue

        selected.append(row.to_dict())
        seen_texts.add(normalized)
        per_doc_count[row["document_id"]] = per_doc_count.get(row["document_id"], 0) + 1
        used_words += chunk_words
        if len(selected) >= max_context_chunks:
            break

    blocks = []
    for i, row in enumerate(selected, start=1):
        label = "CURRENT" if row["is_current"] else "OUTDATED"
        blocks.append(f"[Source {i}] {row['title']} | {row['effective_date']} | {label}\n{row['chunk_text']}")

    return {
        "query": query, "candidates": candidates, "selected": selected,
        "context_text": "\n\n".join(blocks), "num_sources": len(selected), "used_words": used_words,
    }


# ---------------------------------------------------------------------------
# Ground truth + evaluation
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    # --- Paraphrase traps: query wording differs from document wording ---
    "How do I get a handle on my monthly money habits?": [0],
    "What expenses can I actually cut if I'm spending too much?": [2],
    "How can I save money without having to think about it every month?": [4],
    "What are the signs that someone stole my identity?": [13],
    "Can a debt collector legally call me in the middle of the night?": [11],
    "Do banks still charge a lot if I spend more than what's in my account?": [12, 18],

    # --- Direct-wording queries ---
    "How do I create a budget that actually works?": [0],
    "How do I dispute an error on my credit report?": [6],
    "How do I get a free copy of my credit report?": [7, 17],
    "How do I get and keep a good credit score?": [8],
    "Where can I check my credit score for free?": [9],
    "How much should I keep in an emergency fund to start?": [10],
    "When are debt collectors allowed to call me?": [11],
    "How much will I be charged if I overdraft my account?": [12, 18],
    "Why do I keep running out of money before payday?": [14],
    "How can I lower my grocery bill?": [15],
    "How can I spend less on my daily commute?": [16],

    # --- Exact-detail queries ---
    "What percentage of my credit limit should I stay under for a good score?": [8],
    "What time of day are debt collectors not allowed to call me?": [11],

    # --- Multi-document query ---
    "I can't cover all my bills this month, what should I look at first?": [14, 2, 0],
}


def precision_at_k(retrieved_ids, relevant_ids, k):
    hits = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(hits) / k


def recall_at_k(retrieved_ids, relevant_ids, k):
    hits = set(retrieved_ids[:k]) & set(relevant_ids)
    return len(hits) / len(relevant_ids)


def evaluate_context_building(ground_truth: dict, k: int = 3) -> pd.DataFrame:
    rows = []
    for query, relevant_ids in ground_truth.items():
        package = build_context_package(query, retrieval_k=8, max_context_chunks=k)
        retrieved_ids = [row["document_id"] for row in package["selected"]]
        rows.append({
            "query": query, "relevant_ids": relevant_ids, "retrieved_ids": retrieved_ids,
            f"precision@{k}": precision_at_k(retrieved_ids, relevant_ids, k),
            f"recall@{k}": recall_at_k(retrieved_ids, relevant_ids, k),
            "num_sources_used": package["num_sources"],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Evaluating context building over the ground-truth query set ===")
    eval_df = evaluate_context_building(GROUND_TRUTH, k=3)
    print(eval_df.drop(columns=["relevant_ids", "retrieved_ids"]).describe())
    print(f"\nAverage precision@3: {eval_df['precision@3'].mean():.2f}")
    print(f"Average recall@3: {eval_df['recall@3'].mean():.2f}")

    print("\n=== Queries with recall@3 == 0 (worth inspecting) ===")
    print(eval_df[eval_df["recall@3"] == 0][["query", "relevant_ids", "retrieved_ids"]])

    print("\n=== Example: current-vs-outdated conflict, raw candidates vs final context ===")
    conflict_query = "How do I get a free copy of my credit report?"
    raw = retrieve_hybrid(conflict_query, k=5)
    print(raw[["document_id", "title", "effective_date", "is_current", "score"]])
    package = build_context_package(conflict_query)
    print("\nFinal context sent to the LLM:")
    print(package["context_text"])

    print("\n=== Example: out-of-scope query (nothing in the corpus should match well) ===")
    out_of_scope = build_context_package("What is the interest rate on a 30-year fixed mortgage right now?")
    print(f"Sources selected: {out_of_scope['num_sources']} (0 is correct behavior here)")
