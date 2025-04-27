import torch
import torch.nn as nn
import torch.nn.functional as F

class RotationAttack(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u     = nn.Parameter(torch.randn(dim))
        self.v     = nn.Parameter(torch.randn(dim))
        self.theta = nn.Parameter(torch.tensor(0.4))

    def forward(self, x):
        x = x.squeeze(0)
        u = F.normalize(self.u, dim=0)
        v = F.normalize(self.v - torch.dot(u, self.v) * u, dim=0)
        pu = torch.dot(x, u) * u
        pv = torch.dot(x, v) * v
        rest = x - pu - pv
        rot  = torch.cos(self.theta) * pu + torch.sin(self.theta) * pv + rest
        return rot.unsqueeze(0)
