"""
word_importance.py

Implements Phase 1: Word Importance Ranking.
Each word is masked in turn and the change in classifier prediction is measured.
"""

from utils.text_utils import mask_word

def get_important_words(sentence: str, clf, top_k: int = 5):
    """
    Returns a list of the top-k words (with their index and importance score)
    with the highest impact on the classifier’s prediction.
    """
    words = sentence.split()
    original_pred = clf.predict(sentence)[0]
    orig_score = original_pred["score"]

    importance_list = []
    for index, word in enumerate(words):
        masked_sentence = mask_word(sentence, index)
        new_pred = clf.predict(masked_sentence)[0]
        # If prediction label remains the same, compute the score difference; if flipped, use full diff.
        if new_pred["label"] == original_pred["label"]:
            score_diff = orig_score - new_pred["score"]
        else:
            score_diff = orig_score

        importance_list.append({
            "index": index,
            "word": word,
            "importance": score_diff
        })

    # Sort words by importance score in descending order and return the top-k
    sorted_words = sorted(importance_list, key=lambda x: x["importance"], reverse=True)
    return sorted_words[:top_k]
