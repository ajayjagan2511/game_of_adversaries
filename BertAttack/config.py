# config.py
# Global configuration parameters for the adversarial attack project

TOP_K = 3  # Number of top important words to consider for attack
MIN_COSINE_THRESHOLD_WORD = 0.45  # Minimum cosine similarity for word-level semantic preservation
MIN_COSINE_THRESHOLD_SENTENCE = 0.0  # Minimum cosine similarity for sentence-level semantic preservation
MAX_WORD_PERTURBATION_PERCENT = 1.0  # Maximum percentage of words allowed to be perturbed
M_ATTEMPTS = 100  # Maximum attempts for generating a new attack vector if stored ones fail
DIRECTIONAL_THRESHOLD = 0.0  # Minimum required directional effectiveness (placeholder value)

# Model names and paths:
MLM_MODEL_NAME = "bert-base-uncased"
SENTIMENT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
