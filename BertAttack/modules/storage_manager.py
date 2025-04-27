"""
storage_manager.py

Implements Phase 3: Storage Mechanism Implementation.
Stores successful attack vectors for reuse in future attacks.
"""

class StorageManager:
    def __init__(self):
        # Use a simple dictionary (keyed by the original word) for storage.
        self.storage = {}

    def store_attack_vector(self, attack_info: dict):
        """
        Store the attack vector information.
        attack_info should contain:
          - original_word
          - context_index
          - replacement
          - direction_vector (if computed)
          - magnitude (if computed)
          - additional features (e.g. POS tag)
        """
        key = attack_info["original_word"]
        if key not in self.storage:
            self.storage[key] = []
        self.storage[key].append(attack_info)
        print(f"Stored attack vector for '{key}'.")

    def retrieve_attack_vector(self, word: str):
        """
        Retrieve stored attack vectors for the given word.
        For simplicity, return the list of stored vectors.
        """
        if word in self.storage and self.storage[word]:
            return self.storage[word]
        return None
