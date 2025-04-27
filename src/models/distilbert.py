from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_distilbert(device: str = "cpu"):
    hf = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    ).to(device)

    for p in hf.parameters():
        p.requires_grad = False

    tok = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    enc = hf.bert if hasattr(hf, "bert") else hf.distilbert

    return {
        "type": "hf",
        "model": hf,
        "tokenizer": tok,
        "encoder": enc,
    }
