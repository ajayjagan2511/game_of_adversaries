#!/usr/bin/env python3
from datasets import load_dataset
from config import TOP_K
from modules import word_importance, storage_manager, constraints_checker
from models import classifier, pos_tagger

# This evaluation focuses solely on reusing stored attacks.
clf = classifier.SentimentClassifier()
ptagger = pos_tagger.POSTagger()
storage = storage_manager.StorageManager()

def run_storage_only_attack(sentence: str, target_label: str):
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
        stored_vectors = storage.retrieve_attack_vector(original_word)
        if stored_vectors:
            new_candidate = stored_vectors[0]["replacement"]
            if constraints_checker.check_pos_preservation(words[index], new_candidate, ptagger) and \
               constraints_checker.check_word_semantic_similarity(words[index], new_candidate):
                adversarial_sentence[index] = new_candidate
                modifications += 1
                print(f"Replaced '{original_word}' with '{new_candidate}' using storage at index {index}")
        else:
            print(f"No stored vector for '{original_word}' - skipping.")
        if modifications / len(words) > __import__('config').config.MAX_WORD_PERTURBATION_PERCENT:
            print("Maximum perturbation reached.")
            break
    final_sentence = " ".join(adversarial_sentence)
    final_prediction = clf.predict(final_sentence)[0]
    print(f"\nAdversarial Sentence (Storage-only): {final_sentence}")
    print(f"Final Prediction: {final_prediction}")

if __name__ == "__main__":
    # Load 50 samples from the SST-2 validation set for evaluation
    dataset = load_dataset("glue", "sst2", split="validation[:50]")
    for data in dataset:
        sentence = data["sentence"].strip()
        original = clf.predict(sentence)[0]["label"]
        target = "NEGATIVE" if original == "POSITIVE" else "POSITIVE"
        run_storage_only_attack(sentence, target)
