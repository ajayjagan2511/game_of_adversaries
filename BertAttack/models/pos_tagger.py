"""
pos_tagger.py

Uses spaCy to perform part-of-speech tagging.
"""

import spacy

class POSTagger:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download
            download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def get_pos_from_sentence(self, sentence: str, index: int) -> str:
        """
        Returns the POS tag of the word at the given index in the sentence.
        """
        doc = self.nlp(sentence)
        if index < 0 or index >= len(doc):
            return ""
        return doc[index].pos_

    def get_pos(self, word: str) -> str:
        """
        Returns the POS tag for a single word.
        """
        doc = self.nlp(word)
        if doc:
            return doc[0].pos_
        return ""
