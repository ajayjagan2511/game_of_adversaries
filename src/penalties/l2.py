import torch

def l2_pen(o, a):
    return torch.norm(a - o, p=2) ** 2
