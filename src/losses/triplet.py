import torch.nn as nn

_triplet = nn.TripletMarginLoss(margin=5.0)

def triplet_loss(o, a, y):
    from __main__ import curr_model, neg_means
    import torch
    neg = neg_means[curr_model][1 - y.item()].to(o.device) 
    return _triplet(a, neg, o), curr_model.classifier(a)