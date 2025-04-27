"""
text_utils.py

Provides helper functions for text manipulation.
"""

def mask_word(sentence: str, index: int, mask_token: str = "[MASK]") -> str:
    """
    Returns the sentence after replacing the word at the given index with the mask token.
    """
    words = sentence.split()
    if index < 0 or index >= len(words):
        return sentence
    words[index] = mask_token
    return " ".join(words)
