"""
classifier.py

Wrapper for a sentiment classifier using the Hugging Face transformers pipeline.
"""

from transformers import pipeline
from config import SENTIMENT_MODEL_NAME

class SentimentClassifier:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = SENTIMENT_MODEL_NAME
        self.model = pipeline("sentiment-analysis", model=model_name)

    def predict(self, sentence: str) -> list:
        """
        Returns a list of predictions.
        Each prediction is a dict with keys 'label' and 'score'.
        """
        prediction = self.model(sentence)
        return prediction
