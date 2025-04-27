"""
constraints_checker.py

Provides functions to check for:
  - Word-level semantic similarity
  - Sentence-level semantic similarity
  - POS preservation
"""

from utils.embeddings import get_embedding, cosine_similarity

def check_word_semantic_similarity(original_word: str, candidate_word: str, threshold: float = 0.7) -> bool:
    """
    Check whether the candidate word is semantically similar to the original word using cosine similarity.
    """
    orig_emb = get_embedding(original_word)
    cand_emb = get_embedding(candidate_word)
    sim = cosine_similarity(orig_emb, cand_emb)
    return sim >= threshold

def check_sentence_semantic_similarity(original_sentence: str, perturbed_sentence: str, threshold: float = 0.8) -> bool:
    """
    Stub implementation for sentence-level semantic similarity.
    In practice, use a sentence encoder to compute similarity.
    """
    return True

def check_pos_preservation(original_word: str, candidate_word: str, pos_tagger) -> bool:
    """
    Ensures the candidate word's POS tag matches that of the original.
    """
    original_pos = pos_tagger.get_pos(original_word)
    candidate_pos = pos_tagger.get_pos(candidate_word)
    return original_pos == candidate_pos
