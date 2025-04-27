"""
mlm.py

Wrapper for a masked language model (MLM) using the fill-mask pipeline.
"""

from transformers import pipeline
from config import MLM_MODEL_NAME

class MLM:
    def __init__(self, mlm_model_name: str = None):
        if mlm_model_name is None:
            mlm_model_name = MLM_MODEL_NAME
        self.model = pipeline("fill-mask", model=mlm_model_name)
        self.mask_token = self.model.tokenizer.mask_token

    def get_mask_candidates(self, sentence: str, index: int, num_candidates: int = 5):
        """
        Masks the word at the given index and returns a list of candidate token predictions.
        Each candidate is a tuple (token_str, score).
        """
        words = sentence.split()
        if index < 0 or index >= len(words):
            return []
        words[index] = self.mask_token
        masked_sentence = " ".join(words)
        try:
            predictions = self.model(masked_sentence, top_k = num_candidates)
        except Exception as e:
            print(f"MLM error: {e}")
            return []
        candidates = []
        for pred in predictions:
            token_str = pred["token_str"].strip()
            score = pred["score"]
            candidates.append((token_str, score))
        return candidates
