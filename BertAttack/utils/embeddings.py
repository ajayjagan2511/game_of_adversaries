"""
embeddings.py

Provides utilities for computing embeddings and cosine similarities.
Uses SentenceTransformers for embedding generation.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Load the embedding model once globally
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> np.ndarray:
    """
    Returns the embedding vector for the given text.
    """
    emb = embedding_model.encode(text)
    return np.array(emb)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.
    """
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8
    return dot_product / norm_product
