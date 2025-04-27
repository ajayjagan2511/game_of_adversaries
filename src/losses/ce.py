import torch.nn.functional as F

def ce_loss(o, a, y):
    from __main__ import curr_model
    logits = curr_model.classifier(a)
    return F.cross_entropy(logits, y), logits
