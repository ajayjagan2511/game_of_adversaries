import torch

class DeepFoolAttack:
    def __init__(self, steps, overshoot=0.02):
        self.steps     = steps
        self.overshoot = overshoot

    def __call__(self, orig):
        x_adv = orig.clone().detach().requires_grad_(True)
        for _ in range(self.steps):
            from __main__ import curr_model 
            logits = curr_model.classifier(x_adv)
            o_lbl  = logits.argmax(dim=-1)
            vals, idxs = torch.sort(logits, descending=True)
            sec = idxs[0, 1]
            curr_model.zero_grad()
            diff = logits[0, o_lbl] - logits[0, sec]
            diff.backward()
            w = x_adv.grad.data
            r = (abs(diff.item()) + 1e-4) * w / (w.norm()**2)
            x_adv = (x_adv + (1 + self.overshoot) * r).detach().requires_grad_(True)
            if curr_model.classifier(x_adv).argmax(dim=-1) != o_lbl:
                break
        return x_adv.detach()
