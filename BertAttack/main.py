#!/usr/bin/env python3
"""
main.py

This script runs the adversarial text attack pipeline:
  1. Identify the most important words (Phase 1).
  2. Generate context-aware, directional replacements (Phase 2) using up to M_ATTEMPTS tries 
     per word until the candidate either changes the predicted sentiment or exhausts all attempts.
  3. Store and reuse successful attack vectors (Phase 3).

When a replacement candidate is found that successfully flips the sentiment,
the algorithm immediately updates & exits, returning the new sentence.
"""

import config
from config import (
    TOP_K,
    MIN_COSINE_THRESHOLD_WORD,
    MIN_COSINE_THRESHOLD_SENTENCE,  # (if needed for sentence-level checks)
    MAX_WORD_PERTURBATION_PERCENT,
    M_ATTEMPTS,
    DIRECTIONAL_THRESHOLD,
    MLM_MODEL_NAME,
    SENTIMENT_MODEL_NAME,
)
from modules import word_importance, directional_replacement, storage_manager, constraints_checker
from models import classifier, mlm, pos_tagger

# Initialize models and storage using configuration parameters
print("Initializing models ...")
clf = classifier.SentimentClassifier(model_name=SENTIMENT_MODEL_NAME)  # Sentiment classifier
mlm_model = mlm.MLM(mlm_model_name=MLM_MODEL_NAME)                      # Masked Language Model
ptagger = pos_tagger.POSTagger()
storage = storage_manager.StorageManager()

def run_attack(sentence: str, target_label: str):
    print(f"\nOriginal Sentence: {sentence}")
    original_prediction = clf.predict(sentence)[0]
    print(f"Original Prediction: {original_prediction}")

    # Phase 1: Word Importance Ranking using TOP_K from config
    important_words = word_importance.get_important_words(sentence, clf, top_k=TOP_K)
    print(f"\nTop-{TOP_K} important words (index, word, importance):")
    for item in important_words:
        print(item)

    words = sentence.split()
    adversarial_sentence = words.copy()
    modifications = 0
    found_flip = False

    # Phase 2: Iterate over important words to try and find a valid candidate
    for item in important_words:
        index = item["index"]
        original_word = item["word"]

        # First, try to retrieve a stored candidate for this word (if any)
        stored_candidates = storage.retrieve_attack_vector(original_word)
        new_candidate = None
        if stored_candidates:
            new_candidate = stored_candidates[0]["replacement"]
            print(f"Using stored candidate for '{original_word}': {new_candidate}")
        else:
            # Call the directional replacement function which will try up to M_ATTEMPTS.
            new_candidate = directional_replacement.get_directional_replacement(
                sentence=sentence,
                index=index,
                target_label=target_label,
                clf=clf,
                mlm_model=mlm_model,
                ptagger=ptagger,
                directional_threshold=DIRECTIONAL_THRESHOLD,
                m_attempts=M_ATTEMPTS
            )
            if new_candidate:
                # Store the successful candidate for future reuse.
                storage.store_attack_vector({
                    "original_word": original_word,
                    "context_index": index,
                    "replacement": new_candidate,
                    "direction_vector": None,   # Placeholder if computed later
                    "magnitude": None,          # Placeholder if computed later
                    "features": {"pos": ptagger.get_pos_from_sentence(sentence, index)}
                })

        # If a candidate was found, update the sentence and check if sentiment flipped.
        if new_candidate is not None:
            print(f"Replacing '{original_word}' with '{new_candidate}' at index {index}.")
            adversarial_sentence[index] = new_candidate

            # Form the new sentence
            new_sentence = " ".join(adversarial_sentence)
            new_pred = clf.predict(new_sentence)[0]
            print(f"New Sentence: {new_sentence}")
            print(f"New Prediction: {new_pred}")

            # If the sentiment is now flipped to the target, exit early.
            if new_pred["label"] == target_label:
                print(f"Sentiment flipped to {target_label}. Exiting replacement loop.")
                found_flip = True
                break

            modifications += 1

        # Optional: Control overall perturbation (this example allows full perturbation)
        if modifications / len(words) > MAX_WORD_PERTURBATION_PERCENT:
            print("Reached maximum allowed perturbation limit.")
            break

    final_sentence = " ".join(adversarial_sentence)
    new_prediction = clf.predict(final_sentence)[0]
    print(f"\nFinal Adversarial Sentence: {final_sentence}")
    print(f"Final Prediction: {new_prediction}")

    return final_sentence

if __name__ == "__main__":
    test_sentence = "He had a sour attitude, that made him seem like a bad person, which is true."     
    # Determine the target sentiment by flipping the original prediction.
    original_label = clf.predict(test_sentence)[0]["label"]
    target = "NEGATIVE" if original_label == "POSITIVE" else "POSITIVE"
    run_attack(test_sentence, target)
