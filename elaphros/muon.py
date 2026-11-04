"""Muon (Jordan et al. 2024; Liu et al. 2025): momentum SGD whose update for each weight matrix is
orthogonalised by five Newton-Schulz iterations. With adjust='match_rms_adamw' the update is scaled by
0.2 * sqrt(max(rows, cols)) so its RMS matches AdamW and AdamW-scale learning rates apply."""
import torch


def _ns5(G, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    tr = X.size(0) > X.size(1)
    if tr:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return (X.T if tr else X).to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=8e-4, weight_decay=0.1, momentum=0.95, nesterov=True, ns_steps=5,
                 adjust="match_rms_adamw"):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                                      nesterov=nesterov, ns_steps=ns_steps, adjust=adjust))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"].mul_(g["momentum"]).add_(p.grad)
                upd = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
                shape = upd.shape
                U = _ns5(upd.reshape(shape[0], -1).float(), g["ns_steps"])
                rows, cols = U.shape
                if g["adjust"] == "match_rms_adamw":
                    U = U * 0.2 * max(rows, cols) ** 0.5
                else:
                    U = U * max(1.0, rows / cols) ** 0.5
                p.mul_(1 - g["lr"] * g["weight_decay"])
                p.add_(U.reshape(shape).to(p.dtype), alpha=-g["lr"])
