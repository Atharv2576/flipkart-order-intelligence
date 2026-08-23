"""Sentence-wise chunking. A policy sentence is the natural unit of a
policy answer -- "apparel may be returned within 15 days" is a complete,
quotable rule -- so fixed-size or overlapping windows would cut a rule in
half, and multi-sentence chunks would bundle an unrelated rule into every
retrieval hit. Every chunk keeps a pointer back to its parent document,
which is what makes document-level retrieval evaluation possible.
"""
import re

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]


def chunk_document(document: dict) -> list[dict]:
    sentences = split_sentences(document["text"])
    return [
        {
            "chunk_id": f"{document['id']}::s{i:02d}",
            "document_id": document["id"],
            "document_title": document["title"],
            "chunk_index": i,
            "chunk_text": sentence,
        }
        for i, sentence in enumerate(sentences)
    ]


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []
    for document in documents:
        chunks.extend(chunk_document(document))
    return chunks
