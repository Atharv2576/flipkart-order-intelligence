"""A single cached local sentence-transformer encoder, shared by index
building, retrieval, and few-shot intent routing so every embedding in the
system lives in the same vector space.
"""
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        # Forced to CPU deliberately: the agent process loads this encoder
        # alongside a separate torch model (Part 2's classifier) and runs
        # graph nodes through LangGraph's executor, which does not guarantee
        # the same thread for every node. PyTorch's MPS backend is not
        # thread-safe across that pattern and segfaults; a single small
        # embedding model on CPU is fast enough that MPS buys nothing here
        # (MPS is used where it matters -- Part 2's actual training loop).
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalised embeddings, so a FAISS inner-product index computes
    cosine similarity directly.
    """
    encoder = get_encoder()
    vectors = encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")
