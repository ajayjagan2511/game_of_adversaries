#!/usr/bin/env python3
import random
from datasets import load_dataset
from config import TOP_K
from modules import word_importance, directional_replacement, storage_manager, constraints_checker
from models import classifier, mlm, pos_tagger

# Initialize models and storage
clf = classifier.SentimentClassifier()
mlm_model = mlm.MLM()
ptagger = pos_tagger.POSTagger()
storage = storage_manager.StorageManager()

def run_attack(sentence: str, target_label: str):
    print(f"\nOriginal Sentence: {sentence}")
    original_prediction = clf.predict(sentence)[0]
    print(f"Original Prediction: {original_prediction}")
    words = sentence.split()
    important_words = word_importance.get_important_words(sentence, clf, top_k=TOP_K)
    modifications = 0
    adversarial_sentence = words.copy()
    for item in important_words:
        index = item["index"]
        original_word = item["word"]
        # Try to use a stored attack vector first
        stored_vectors = storage.retrieve_attack_vector(original_word)
        new_candidate = None
        if stored_vectors:
            new_candidate = stored_vectors[0]["replacement"]
            print(f"Using stored vector for '{original_word}' -> '{new_candidate}'")
        else:
            new_candidate = directional_replacement.get_directional_replacement(
                sentence, index, target_label, clf, mlm_model, ptagger)
            if new_candidate:
                storage.store_attack_vector({
                    "original_word": original_word,
                    "context_index": index,
                    "replacement": new_candidate,
                    "direction_vector": None,
                    "magnitude": None,
                    "features": {"pos": ptagger.get_pos_from_sentence(sentence, index)}
                })
        if new_candidate and constraints_checker.check_pos_preservation(words[index], new_candidate, ptagger) and \
           constraints_checker.check_word_semantic_similarity(words[index], new_candidate):
            adversarial_sentence[index] = new_candidate
            modifications += 1
            print(f"Replaced '{original_word}' with '{new_candidate}' at index {index}")
        if modifications / len(words) > __import__('config').config.MAX_WORD_PERTURBATION_PERCENT:
            print("Maximum perturbation reached.")
            break
    final_sentence = " ".join(adversarial_sentence)
    final_prediction = clf.predict(final_sentence)[0]
    print(f"\nAdversarial Sentence: {final_sentence}")
    print(f"Final Prediction: {final_prediction}")

if __name__ == "__main__":
    # Load 50 samples from the IMDb test set for quick evaluation
    dataset = load_dataset("imdb", split="test[:50]")
    for data in dataset:
        sentence = data["text"].replace("\n", " ").strip()
        original = clf.predict(sentence)[0]["label"]
        target = "NEGATIVE" if original == "POSITIVE" else "POSITIVE"
        run_attack(sentence, target)
