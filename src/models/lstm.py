from textattack.models.helpers.lstm_for_classification import LSTMForClassification

def load_lstm(device: str = "cpu"):
    ta = LSTMForClassification.from_pretrained("lstm-sst2").to(device)

    return {
        "type": "ta",
        "model": ta,
        "tokenizer": ta.tokenizer,
    }
