kappa = 0.0

def cw_loss(o, a, y):
    from __main__ import curr_model
    logits       = curr_model.classifier(a)
    target_label = y.item()         
    orig_label   = 1 - target_label  

    target_logit = logits[0, target_label]
    orig_logit   = logits[0, orig_label]

    loss = orig_logit - target_logit + kappa  
    return loss, logits

