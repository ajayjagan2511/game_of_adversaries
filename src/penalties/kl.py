import torch.nn.functional as F
import torch

def kl_pen(o, a):
    from __main__ import curr_model
    with torch.no_grad():
        p_clean = F.softmax(curr_model.classifier(o), dim=-1)
    p_adv_log = F.log_softmax(curr_model.classifier(a), dim=-1)
    return F.kl_div(p_adv_log, p_clean, reduction="batchmean")
