class PGDAttack:
    # unchanged logic, just moved
    def __init__(self, alpha, steps, lam):
        self.alpha = alpha
        self.steps = steps
        self.lam   = lam

    def __call__(self, orig, loss_fn, pen_fn, y):
        import torch
        x_adv = orig.clone().detach().requires_grad_(True)
        for _ in range(self.steps):
            beta = 1 if loss_fn.__name__ in ('ce_loss', 'cw_loss') else -1
            clf, _ = loss_fn(orig, x_adv, y)
            pen    = pen_fn(orig, x_adv)
            obj    = clf + beta * self.lam * pen
            obj.backward()
            with torch.no_grad():
                step = self.alpha * x_adv.grad.sign()
                if loss_fn.__name__ in ('ce_loss', 'cw_loss'):
                    x_adv = x_adv - step
                else:
                    x_adv = x_adv + step
                x_adv = x_adv.detach().requires_grad_(True)
        return x_adv.detach()
