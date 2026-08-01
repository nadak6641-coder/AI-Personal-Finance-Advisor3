"""
02_preprocessing.py
====================
Cleans and normalizes the raw document text before chunking.

Two normalization variants are kept separate on purpose:
  - clean_display_text: light cleanup, preserves readability. This is what
    gets stored and eventually shown to the user as a cited source.
  - normalize_lexical_text: aggressive cleanup (lowercase, strip
    punctuation). This is only used internally for lexical/keyword
    matching later in the pipeline, never shown to the user.
"""

import re

from importlib import import_module

_documents_module = import_module("01_documents")
DOCUMENTS = _documents_module.DOCUMENTS


def clean_display_text(text: str) -> str:
    """Light cleanup: collapse whitespace, strip leading/trailing spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_lexical_text(text: str) -> str:
    """Aggressive cleanup for keyword-based matching only."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_documents(documents: list) -> list:
    """Apply display-level cleaning to every document's text field."""
    cleaned = []
    for doc in documents:
        cleaned_doc = dict(doc)
        cleaned_doc["text"] = clean_display_text(doc["text"])
        cleaned.append(cleaned_doc)
    return cleaned


CLEANED_DOCUMENTS = preprocess_documents(DOCUMENTS)


if __name__ == "__main__":
    print(f"Preprocessed {len(CLEANED_DOCUMENTS)} documents")
    print("\nExample (document 0) before/after:")
    print("BEFORE:", repr(DOCUMENTS[0]["text"][:120]))
    print("AFTER: ", repr(CLEANED_DOCUMENTS[0]["text"][:120]))
