import torch.nn.functional as F

def cos_pen(o, a):
    return 1 - F.cosine_similarity(a, o)
