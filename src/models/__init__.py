
from .distilbert import load_distilbert
from .lstm       import load_lstm

def load_model(name: str, device: str):

    if name == "DistilBERT":
        return load_distilbert(device)
    if name == "LSTM":
        return load_lstm(device)
    raise ValueError(f"Unknown model '{name}'")
